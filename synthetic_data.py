
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SVMonteCarlo:
  def __init__(self, S_0, r, rho, N_t, dt, M, t_samples, k_samples, v_samples, device):
    '''
    S_0 : torch tensor, starting asset price
    r : torch tensor, risk free rate
    rho : torch tensor, correlation between asset price and volatility
    N_t : int, number of time steps
    dt : torch tensor, time step size
    M : int, number of paths
    t_samples : int, number of time samples
    k_samples : int, number of strike samples
    v_samples : list, list of volatility samples
    '''
    self.S_0 = S_0
    self.r = r
    self.rho = rho
    self.N_t = N_t
    self.dt = dt
    self.M = M
    self.t_samples = t_samples
    self.k_samples = k_samples
    self.v_samples = v_samples
    self.device = device
    self.data_type = torch.float32

  def exact_alpha(self, K, T, v):
    result = v * (torch.tensor(1.0) - torch.tensor(10.0) * v)
    return result

  def exact_beta(self, K, T, v):
    return torch.tensor(2.0) * v**(1.5)

  def perform_monte_carlo(self, vol):
    t_all = torch.reshape(torch.tensor(np.linspace(0, (self.N_t*self.dt).item(), self.N_t), dtype=self.data_type).to(self.device), (-1,1))
    S_list = [torch.full((1, self.M), self.S_0[0].item(), dtype=self.data_type, device=self.device)]
    v_list = [torch.full((1, self.M), vol, dtype=self.data_type, device=self.device)]
    Z_1_mat = torch.randn(self.N_t, self.M, dtype=self.data_type, device=self.device) * torch.sqrt(self.dt)
    Z_2_mat = torch.randn(self.N_t, self.M, dtype=self.data_type, device=self.device) * torch.sqrt(self.dt)
    dW_1_list = Z_1_mat
    dW_2_list = self.rho * Z_1_mat + torch.sqrt(1-self.rho**2) * Z_2_mat

    for i in range(self.N_t - 1):
      t_now = t_all[i]
      S_now = S_list[-1]
      v_now = v_list[-1]
      S_new = S_now + self.r * S_now * self.dt + torch.sqrt(v_now) * S_now * dW_1_list[i]
      v_new = torch.nn.functional.relu(v_now + self.exact_alpha(S_now, t_now, v_now) * self.dt + self.exact_beta(S_now, t_now, v_now) * torch.sqrt(v_now) * dW_2_list[i])
      S_list.append(S_new)
      v_list.append(v_new)
    return S_list, v_list, t_all

  def plot_asset_trajectories(self):
    fig, ax = plt.subplots(figsize=[12, 3], dpi=450)
    plt.plot(self.t_all.cpu().numpy(), self.S_matrix[:,:1024].cpu().numpy(), lw=0.1)
    plt.title('Asset Price Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.show()
    plt.close()

  def plot_volatility_trajectories(self):
    fig, ax = plt.subplots(figsize=[12, 3], dpi=450)
    plt.plot(self.t_all.cpu().numpy(), self.v_matrix[:,:1024].cpu().numpy(), lw=0.1)
    plt.title('Volatility Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.show()
    plt.close()

  def synthetic_data(self, S_matrix, vol):
    step = max(1, 120 // self.t_samples)
    time_indices = list(range(30, self.N_t, step))
    times = self.t_all[30::(120 // self.t_samples)].squeeze().tolist()
    strikes = np.linspace(500, 2000, self.k_samples)
    rows = []

    for i, t_idx in enumerate(time_indices):
      time = times[i]

      S_at_t = S_matrix[t_idx, :].cpu().numpy()

      for strike in strikes:
        payoff = np.maximum(S_at_t - strike, 0)
        exp_payoff = np.mean(payoff)
        price = np.exp(-self.r * time) * exp_payoff
        rows.append([strike, time, vol, price.item()])

    return rows

  def run(self):
    all_data_rows = []

    print(f'Starting simulation with {self.M} paths...')

    for i, vol in enumerate(self.v_samples):
      S_list, v_list, self.t_all = self.perform_monte_carlo(vol)
      S_matrix = torch.cat(S_list, dim=0)

      if i==0:
        self.S_matrix = S_matrix
        self.v_matrix = torch.cat(v_list, dim=0)
        self.plot_asset_trajectories()
        self.plot_volatility_trajectories()

      data_rows = self.synthetic_data(S_matrix, vol)
      all_data_rows.extend(data_rows)

    self.df = pd.DataFrame(all_data_rows, columns=['Strike', 'Time', 'Volatility', 'Price'])
    print("Synthetic data generation complete.")
    print(self.df.head())
    return
