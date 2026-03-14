
import torch
import numpy as np
import pandas as pd

class MCRepriceError:
    def __init__(self, df, r, q, S_0, rho, alpha, beta, dt, device='cpu'):
        self.df = df.copy()
        self.r = r
        self.q = q
        self.S_0 = S_0 if isinstance(S_0, float) else S_0.item()
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.device = device

        self.T_max = self.df['Time'].max()
        self.data_type = torch.float32

    def run_vectorized(self, n_paths=10000):
        M = len(self.df)
        vols = torch.tensor(self.df['Volatility'].values, dtype=self.data_type, device=self.device).view(M, 1)
        strikes = torch.tensor(self.df['Strike'].values, dtype=self.data_type, device=self.device).view(M, 1)
        maturities = torch.tensor(self.df['Maturity'].values, dtype=self.data_type, device=self.device).view(M, 1)
        prices = torch.tensor(self.df['Price'].values, dtype=self.data_type, device=self.device).view(M, 1)

        maturity_steps = torch.clamp((maturities / self.dt).long(), min=2)
        max_steps = maturity_steps.max().item()

        S_now = torch.full((M, n_paths), self.S_0, dtype=self.data_type, device=self.device)
        v_now = vols.expand(M, n_paths).clone()

        S_at_maturity = torch.zeros((M, n_paths), dtype=self.data_type, device=self.device)

        dt_sqrt = np.sqrt(self.dt)
        drift_term = (self.r - self.q) * self.dt

        self.alpha.eval()
        self.beta.eval()

        print(f"Starting Vectorized Monte Carlo: {M} options, {n_paths} paths, max steps: {max_steps}...")

        with torch.no_grad():
            for i in range(1, max_steps + 1):
                t_now = i * self.dt
                scaled_t = t_now / self.T_max

                Z_1 = torch.randn(M, n_paths, dtype=self.data_type, device=self.device) * dt_sqrt
                Z_2 = torch.randn(M, n_paths, dtype=self.data_type, device=self.device) * dt_sqrt
                dW_1 = Z_1
                dW_2 = self.rho * Z_1 + np.sqrt(1 - self.rho**2) * Z_2

                t_expanded = torch.full((M * n_paths, 1), scaled_t, dtype=self.data_type, device=self.device)
                v_flat = v_now.reshape(-1, 1)

                net_input = torch.cat((t_expanded, v_flat), dim=1)

                alpha_pred = self.alpha(net_input).view(M, n_paths)
                beta_pred = self.beta(net_input).view(M, n_paths)

                S_new = S_now + drift_term * S_now + torch.sqrt(v_now) * S_now * dW_1
                v_new = torch.relu(v_now + alpha_pred * self.dt + beta_pred * torch.sqrt(v_now) * dW_2)

                S_now = S_new
                v_now = v_new

                mask = (maturity_steps == i)
                S_at_maturity = torch.where(mask, S_now, S_at_maturity)

        payoffs = torch.relu(S_at_maturity - strikes)
        expected_payoffs = payoffs.mean(dim=1, keepdim=True)

        discount_factors = torch.exp(-self.r * maturities)
        reprice_vals = discount_factors * expected_payoffs

        reprice_np = reprice_vals.cpu().numpy().flatten()
        self.df['reprice'] = reprice_np
        self.df['abs_pct_error'] = ((self.df['reprice'] - self.df['Price']) / self.df['Price']).abs() * 100

        mean_reprice_error = self.df['abs_pct_error'].mean()

        print("\n--- Calibration Results ---")
        print(self.df[['Strike', 'Matruity', 'Price', 'reprice', 'abs_pct_error']].head())
        print(f"\nMean Reprice Error (MAPE): {mean_reprice_error:.2f}%")

        return mean_reprice_error
