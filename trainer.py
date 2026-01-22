import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

class TrainSV():
  def __init__(self, S_0, r, rho, nn_params, df, lambda_arb, lambda_pde, lambda_bound, lambda_reg, num_epochs, learning_rate, device):
    self.S_0 = S_0
    self.r =  r
    self.rho = rho
    self.df = df
    self.lambda_arb = lambda_arb
    self.lambda_bound = lambda_bound
    self.lambda_pde = lambda_pde
    self.lambda_reg = lambda_reg
    self.num_epochs = num_epochs
    self.device = device

    self.K_max = df['Strike'].max()
    self.T_max = df['Time'].max()
    self.T_min = df['Time'].min()
    self.v_max = df['Volatility'].max()
    self.v_min = df['Volatility'].min()

    self.scaled_time = df['Time'] / self.T_max
    self.scaled_strike = (np.exp(-r * df['Time']) / self.K_max) * df['Strike']
    self.u = np.log(self.scaled_strike)
    self.u_min = self.u.min()
    self.df_scaled = pd.DataFrame({
        'Strike': self.u,
        'Time': self.scaled_time,
        'Volatility': df['Volatility'],
        'Price': df['Price']
    })
    self.dataset = options_dataset(self.df_scaled, device=self.device)
    self.dataset_unscaled = options_dataset(df, device=self.device)

    self.nn_params = nn_params
    self.NN_call = NN_Call(self.nn_params).to(self.device)
    self.NN_alpha = NetAlpha(self.nn_params).to(self.device)
    self.NN_beta = NetBeta(self.nn_params).to(self.device)

    self.learning_rate = learning_rate
    self.optimizer_NN_call = optim.Adam(self.NN_call.parameters(), lr=self.learning_rate)
    self.optimizer_NN_alpha = optim.Adam(self.NN_alpha.parameters(), lr=self.learning_rate / 4)
    self.optimizer_NN_beta = optim.Adam(self.NN_beta.parameters(), lr=self.learning_rate / 4)

  def call(self, x):
    # x = u, t, v
    return self.S_0 * (1 - torch.exp(-(1-torch.exp(x[:,0])) * self.NN_call(x)))

  def true_alpha(self, x):
    result = torch.tensor(1.5) * (torch.tensor(0.5) - x[:, 2])
    return result

  def true_beta(self, x):
    return torch.tensor(0.7)

  def loss_data(self):
    call_pred = self.call(self.dataset.x)
    call_true = self.dataset.price
    with torch.no_grad():
      weight_ = torch.clamp(torch.mean(torch.abs(call_true)) / torch.abs(call_true), min=0.1, max=10)
    loss_data = torch.mean(weight_ * (call_pred - call_true) ** 2)
    return loss_data

  def loss_bound(self):
    t_zeros = torch.zeros((128,1), dtype = torch.float32).to(self.device)
    v_zeros = torch.zeros((128,1), dtype = torch.float32, requires_grad = True).to(self.device)
    u_rand = -self.u_min * torch.rand((128,1), dtype = torch.float32, device = self.device) + self.u_min
    t_rand = torch.rand((128,1), dtype = torch.float32, device = self.device)
    v_rand = (self.v_max - self.v_min) * torch.rand((128,1), dtype = torch.float32, device = self.device) + self.v_min

    t_rand.requires_grad = True

    x_rand_t0 = torch.cat((u_rand, t_zeros, v_rand), dim = 1)
    x_rand_v0 = torch.cat((u_rand, t_rand, v_zeros), dim = 1)

    call_true_t0 = torch.nn.functional.relu(self.S_0 - self.K_max * torch.exp(u_rand))
    call_true_v0 = torch.nn.functional.relu(self.S_0 - self.K_max * torch.exp(u_rand) * torch.exp(-self.r * self.T_max * t_rand), 0)

    call_pred_t0 = self.call(x_rand_t0)
    call_pred_v0 = self.call(x_rand_v0)
    alpha_pred = self.NN_alpha(x_rand_v0)

    grad_call_v0_pred_t = torch.autograd.grad(call_pred_v0, t_rand, grad_outputs=torch.ones_like(call_pred_v0), create_graph=True)[0]
    grad_call_v0_pred_v = torch.autograd.grad(call_pred_v0, v_zeros, grad_outputs=torch.ones_like(call_pred_v0), create_graph=True)[0]

    with torch.no_grad():
        weight_t = torch.clamp(torch.mean(torch.abs(call_true_t0)) / torch.abs(call_true_t0), min=0.1, max=10)
        weight_v = torch.clamp(torch.mean(torch.abs(call_true_v0)) / torch.abs(call_true_v0), min=0.1, max=10)
        weight_dt = torch.clamp(torch.mean(torch.abs(grad_call_v0_pred_t)) / torch.abs(grad_call_v0_pred_t), min=0.1, max=10)

    v0_pde = (1 / self.T_max) * grad_call_v0_pred_t + alpha_pred * grad_call_v0_pred_v - self.r * call_pred_v0

    loss_t0 = torch.mean(weight_t * (call_pred_t0 - call_true_t0)**2)
    loss_v0 = torch.mean(weight_v * (call_pred_v0 - call_true_v0)**2)
    loss_v0_flow = torch.mean(weight_dt * (v0_pde)**2)
    loss_bound = loss_t0 + loss_v0 + loss_v0_flow
    return loss_bound

  def loss_pde(self):
    t_anchored = torch.zeros((256,1), dtype = torch.float32, device = self.device)
    u_anchored = torch.zeros((256,1), dtype = torch.float32, device = self.device)
    v_anchored_0 = torch.zeros((128,1), dtype = torch.float32, device = self.device)
    v_anchored_1 = torch.full((128,1), self.v_max, dtype = torch.float32, device = self.device)

    u_rand_bulk = -self.u_min * torch.rand((64**2,1), dtype = torch.float32, device = self.device) + self.u_min
    t_rand_bulk = torch.rand((64**2, 1), dtype=torch.float32).to(self.device)
    v_rand_bulk = torch.tensor(self.v_max - self.v_min) * torch.rand((64**2,1), dtype = torch.float32, device = self.device) + torch.tensor(self.v_min)

    t_rand = torch.cat([t_anchored, t_rand_bulk], dim=0).to(self.device)
    u_rand = torch.cat([u_anchored, u_rand_bulk], dim=0).to(self.device)
    v_rand = torch.cat([v_anchored_0, v_anchored_1, v_rand_bulk], dim=0).to(self.device)

    t_rand.requires_grad = True
    u_rand.requires_grad = True
    v_rand.requires_grad = True

    x_rand = torch.cat((u_rand, t_rand, v_rand), dim = 1)

    call_pred = self.call(x_rand)
    alpha_pred = self.NN_alpha(x_rand)
    beta_pred = self.NN_beta(x_rand)

    grad_pred_t = torch.autograd.grad(call_pred, t_rand, grad_outputs=torch.ones_like(call_pred), create_graph=True)[0]
    grad_pred_u = torch.autograd.grad(call_pred, u_rand, grad_outputs=torch.ones_like(call_pred), create_graph=True)[0]
    grad_pred_v = torch.autograd.grad(call_pred, v_rand, grad_outputs=torch.ones_like(call_pred), create_graph=True)[0]

    grad_pred_uu = torch.autograd.grad(grad_pred_u, u_rand, grad_outputs=torch.ones_like(grad_pred_u), create_graph=True)[0]
    grad_pred_vv = torch.autograd.grad(grad_pred_v, v_rand, grad_outputs=torch.ones_like(grad_pred_v), create_graph=True)[0]
    grad_pred_uv = torch.autograd.grad(grad_pred_u, v_rand, grad_outputs=torch.ones_like(grad_pred_u), create_graph=True)[0]

    f_pde = (
        (1 / self.T_max) * grad_pred_t
        + 0.5 * v_rand * (grad_pred_uu - grad_pred_u)
        + self.rho * beta_pred * v_rand * grad_pred_uv
        + 0.5 * v_rand * (beta_pred ** 2) * grad_pred_vv
        + alpha_pred * grad_pred_v
        - self.r * call_pred
    )

    f_arb_1 = torch.nn.functional.relu(grad_pred_t - self.r * self.T_max * grad_pred_u)
    f_arb_2 = torch.nn.functional.relu(grad_pred_uu - grad_pred_u)

    with torch.no_grad():
      weight_ = torch.clamp(torch.mean(torch.abs(grad_pred_t)) / torch.abs(grad_pred_t), min=0.1, max=10)

    loss_pde = torch.mean(weight_ * (f_pde)**2)
    loss_arb = torch.mean(weight_ * (torch.nn.functional.relu(-f_arb_1) + torch.nn.functional.relu(-f_arb_2)))
    loss_reg = torch.mean(weight_ * torch.relu(-grad_pred_t * (grad_pred_uu - grad_pred_u)))
    return loss_pde, loss_arb, loss_reg

  def train_step(self):
    self.optimizer_NN_call.zero_grad()
    self.optimizer_NN_alpha.zero_grad()
    self.optimizer_NN_beta.zero_grad()
    loss_bound = self.loss_bound()
    loss_pde, loss_arb, loss_reg = self.loss_pde()
    loss_alpha = self.lambda_pde * loss_pde + self.lambda_bound * loss_bound
    loss_alpha.backward()
    self.optimizer_NN_alpha.step()

    self.optimizer_NN_call.zero_grad()
    self.optimizer_NN_alpha.zero_grad()
    self.optimizer_NN_beta.zero_grad()
    loss_pde, loss_arb, loss_reg = self.loss_pde()
    loss_beta = self.lambda_pde * loss_pde
    loss_beta.backward()
    self.optimizer_NN_beta.step()

    self.optimizer_NN_call.zero_grad()
    self.optimizer_NN_alpha.zero_grad()
    self.optimizer_NN_beta.zero_grad()
    loss_data = self.loss_data()
    loss_bound = self.loss_bound()
    loss_pde, loss_arb, loss_reg = self.loss_pde()
    loss_call = loss_data + self.lambda_arb * loss_arb + self.lambda_pde * loss_pde + self.lambda_bound * loss_bound + self.lambda_reg * loss_reg
    loss_call.backward()
    self.optimizer_NN_call.step()

    return loss_data.item(), loss_bound.item(), loss_pde.item(), loss_arb.item(), loss_reg.item()

  def run(self):

    exact_alpha = self.true_alpha(self.dataset_unscaled.x)
    exact_beta = self.true_beta(self.dataset_unscaled.x)

    best_alpha_beta_error = float('inf')
    model_alpha_save_path = "best_model_alpha.pth"
    model_beta_save_path = "best_model_beta.pth"

    for epoch in range(self.num_epochs):
      loss_data, loss_bound, loss_pde, loss_arb, loss_reg = self.train_step()

      alpha_pred = self.NN_alpha(self.dataset.x)
      beta_pred = self.NN_beta(self.dataset.x)

      alpha_error = torch.mean(torch.abs(alpha_pred - exact_alpha) / torch.abs(exact_alpha))
      beta_error = torch.mean(torch.abs(beta_pred - exact_beta) / exact_beta)
      alpha_beta_error = torch.maximum(alpha_error, beta_error)

      if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}')

        print(f'Data loss, {loss_data}')
        print(f'PDE loss, {loss_pde}')
        print(f'Regularization loss, {loss_reg}')
        print(f'Arbitrage loss, {loss_arb}')
        print(f'Boundary loss, {loss_bound}')
        print(f'Alpha error, {alpha_error}')
        print(f'Beta error, {beta_error}')
        print('--------------------------------------')

      if alpha_beta_error < best_alpha_beta_error:
        best_alpha_beta_error = alpha_beta_error
        torch.save(self.NN_alpha.state_dict(), model_alpha_save_path)
        torch.save(self.NN_beta.state_dict(), model_beta_save_path)

    print(f'Best alpha-beta error {best_alpha_beta_error}')
    print('Best model paths loaded')
    self.NN_alpha.load_state_dict(torch.load(model_alpha_save_path))
    self.NN_beta.load_state_dict(torch.load(model_beta_save_path))

  def plot_trajectories(self, v_0):
    t_all = torch.linspace(0, 1.5, 150, dtype=torch.float32, device=self.device)
    S_now = torch.tensor(self.S_0, dtype=torch.float32, device=self.device)
    v_now = torch.tensor(v_0, dtype=torch.float32, device=self.device)
    S_path_true = [S_now]
    v_path_true = [v_now]
    sqrt_dt = torch.sqrt(torch.tensor(0.01))
    Z1 = torch.randn(150, dtype=torch.float32, device=self.device) * sqrt_dt
    Z2 = torch.randn(150, dtype=torch.float32, device=self.device) * sqrt_dt
    dW1 = Z1
    dW2 = self.rho * Z1 + torch.sqrt(torch.tensor(1 - self.rho**2)) * Z2

    for i in range(150 - 1):
      t_i = t_all[i]

      S_now = S_now + self.r * S_now * 0.01 + torch.sqrt(v_now) * S_now * dW1[i]

      input_tensor = torch.cat([S_now.view(-1), t_i.view(-1), v_now.view(-1)]).unsqueeze(0)

      v_inc = self.true_alpha(input_tensor) * 0.01 + self.true_beta(input_tensor) * torch.sqrt(v_now) * dW2[i]
      v_now = torch.nn.functional.relu(v_now + v_inc)

      S_path_true.append(S_now.clone())
      v_path_true.append(v_now.clone())

    S_now_1 = torch.tensor(self.S_0, dtype=torch.float32, device=self.device)
    v_now_1 = torch.tensor(v_0, dtype=torch.float32, device=self.device)
    S_path_pred = [S_now_1]
    v_path_pred = [v_now_1]

    for i in range(150 - 1):
      t_i = t_all[i]

      S_now_1 = S_now_1 + self.r * S_now_1 * 0.01 + torch.sqrt(v_now_1) * S_now_1 * dW1[i]

      u = torch.log(torch.exp(-self.r * t_i) * S_now_1 / self.K_max)
      t = t_i / self.T_max
      x = torch.cat((u.view(-1), t.view(-1), v_now_1.view(-1)), dim=0).unsqueeze(0)

      v_inc_1 = self.NN_alpha(x) * 0.01 + self.NN_beta(x) * torch.sqrt(v_now_1) * dW2[i]
      v_now_1 = torch.nn.functional.relu(v_now_1 + v_inc_1)

      S_path_pred.append(S_now_1.clone())
      v_path_pred.append(v_now_1.clone())


    S_path_true_np = torch.stack([x.view(-1) for x in S_path_true]).squeeze().cpu().detach().numpy()
    v_path_true_np = torch.stack([x.view(-1) for x in v_path_true]).squeeze().cpu().detach().numpy()

    S_path_pred_np = torch.stack([x.view(-1) for x in S_path_pred]).squeeze().cpu().detach().numpy()
    v_path_pred_np = torch.stack([x.view(-1) for x in v_path_pred]).squeeze().cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=[12, 3], dpi=450)
    plt.plot(t_all.cpu().numpy(), S_path_true_np, lw=0.1, label='SDE with exact alpha and beta', linestyle='--')
    plt.plot(t_all.cpu().numpy(), S_path_pred_np, lw=0.1, label='SDE with Modeled alpha and beta')
    plt.title('Asset Price Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=[12, 3], dpi=450)
    plt.plot(t_all.cpu().numpy(), v_path_true_np, lw=0.1, label='SDE with exact alpha and beta', linestyle='--')
    plt.plot(t_all.cpu().numpy(), v_path_pred_np, lw=0.1, label='SDE with Modeled alpha and beta')
    plt.title('Volatility Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Volatility Price')
    plt.show()
    plt.close()
