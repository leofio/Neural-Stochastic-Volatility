
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
    result = torch.tensor([1.5]) * (torch.tensor([0.5]) - v)
    return result

  def exact_beta(self, K, T, v):
    return torch.tensor([0.7])

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

  def synthetic_data(self):
    times = self.t_all[30::(120 // self.t_samples)].squeeze().tolist()
    strikes = np.linspace(500, 2000, self.k_samples)
    rows = []
    for vol in v_samples:
      S_times = self.paths[vol][30::(120 // self.t_samples)].squeeze().tolist()
      for idx in range(len(times)):
        S = S_times[idx]
        t = times[idx]
        for strike in strikes:
          price = np.exp(-self.r.item() * t) * np.maximum(S - strike, 0)
          row = [strike, t, vol, price]
          rows.append(row)
    self.df = pd.DataFrame(rows, columns=['Strike', 'Time', 'Volatility', 'Price'])
    return self.df

  def run(self):
    S_list, v_list, self.t_all = self.perform_monte_carlo(self.v_samples[0])
    self.S_matrix = torch.cat(S_list, dim=0)
    self.v_matrix = torch.cat(v_list, dim=0)

    self.paths = {}
    self.paths[self.v_samples[0]] = torch.mean(self.S_matrix, dim=1)
    for vol in self.v_samples[1:]:
      S, v, t = self.perform_monte_carlo(vol)
      S_mat = torch.cat(S, dim=0)
      S_av = torch.mean(S_mat, dim=1)
      self.paths[vol] = S_av

    self.plot_asset_trajectories()
    self.plot_volatility_trajectories()
    self.synthetic_data()
    print(self.df)
    return
