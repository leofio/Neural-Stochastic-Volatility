
import torch
import numpy as np
import pandas as pd

class CallSurfaceRepriceError:
    def __init__(self, trained_model, test_df):
        self.model = trained_model
        self.df = test_df.copy()
        self.device = trained_model.device

    def evaluate(self):
        self.model.NN_call.eval()

        with torch.no_grad():
            scaled_time = self.df['Time'] / self.model.T_max

            scaled_strike = (np.exp(-self.model.r * self.df['Time']) / self.model.K_max) * self.df['Strike']
            u_scaled = np.log(scaled_strike)

            u_tensor = torch.tensor(u_scaled.values, dtype=torch.float32).view(-1, 1).to(self.device)
            t_tensor = torch.tensor(scaled_time.values, dtype=torch.float32).view(-1, 1).to(self.device)
            v_tensor = torch.tensor(self.df['Volatility'].values, dtype=torch.float32).view(-1, 1).to(self.device)

            x_input = torch.cat([u_tensor, t_tensor, v_tensor], dim=1)

            pred_scaled_price = self.model.call(x_input).cpu().numpy().flatten()

            pred_price = pred_scaled_price * self.model.S_0

        self.df['Model_Price'] = pred_price
        self.df['Market_Price'] = self.df['Price']

        self.df['Abs_Error'] = np.abs(self.df['Model_Price'] - self.df['Market_Price'])
        self.df['Abs_Pct_Error'] = (self.df['Abs_Error'] / self.df['Market_Price']) * 100

        mae = self.df['Abs_Error'].mean()
        rmse = np.sqrt(((self.df['Model_Price'] - self.df['Market_Price'])**2).mean())
        mape = self.df['Abs_Pct_Error'].mean()

        print("\n--- Neural Network Call Surface Calibration ---")
        print(self.df[['Strike', 'Time', 'Volatility', 'Market_Price', 'Model_Price', 'Abs_Pct_Error']].head())
        print("-----------------------------------------------")
        print(f"Mean Absolute Error (MAE): ${mae:.4f}")
        print(f"Root Mean Square Error (RMSE): ${rmse:.4f}")
        print(f"Mean Abs Percentage Error (MAPE): {mape:.2f}%")

        return self.df, mae, mape
