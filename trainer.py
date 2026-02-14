
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class TrainSV():
  def __init__(self, S_0, r, rho, nn_params, nn_call_params, tol, df, loss_weights, num_epochs, learning_rate, device, use_adaptive_loss_weights=False, use_scheduler = False, use_LBFGS = False):
    self.S_0 = S_0
    self.r =  r
    self.rho = rho
    self.tol = tol
    self.df = df
    self.lambda_data = loss_weights[0]
    self.lambda_arb = loss_weights[1]
    self.lambda_bound = loss_weights[3]
    self.lambda_pde = loss_weights[2]
    self.lambda_reg = loss_weights[4]
    self.use_adaptive_loss_weights = use_adaptive_loss_weights
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
        'Price': df['Price'] / self.S_0
    })
    self.dataset = options_dataset(self.df_scaled, device=self.device)
    self.dataset_unscaled = options_dataset(df, device=self.device)

    self.nn_params = nn_params
    self.nn_call_params = nn_call_params
    self.NN_call = NN_Call(self.nn_call_params).to(self.device)
    self.NN_alpha = NetAlpha(self.nn_params).to(self.device)
    self.NN_beta = NetBeta(self.nn_params).to(self.device)

    self.learning_rate = learning_rate

    self.optimizer_NN_call_Adam = torch.optim.Adam(self.NN_call.parameters(), lr=self.learning_rate)

    self.optimizer_NN_alpha_Adam_phase_1 = torch.optim.Adam(self.NN_alpha.parameters(), lr= 5 * self.learning_rate)
    self.optimizer_NN_beta_Adam_phase_1 = torch.optim.Adam(self.NN_beta.parameters(), lr= 5 * self.learning_rate)

    self.optimizer_NN_alpha_Adam_phase_2 = torch.optim.Adam(self.NN_alpha.parameters(), lr= self.learning_rate)
    self.optimizer_NN_beta_Adam_phase_2 = torch.optim.Adam(self.NN_beta.parameters(), lr= self.learning_rate)

    self.use_LBFGS = use_LBFGS

    if self.use_LBFGS:
      self.optimizer_NN_call_LBFGS = torch.optim.LBFGS(self.NN_call.parameters(), lr=0.1, line_search_fn='strong_wolfe')
      self.optimizer_NN_alpha_LBFGS = torch.optim.LBFGS(self.NN_alpha.parameters(), lr=0.1, line_search_fn='strong_wolfe')
      self.optimizer_NN_beta_LBFGS = torch.optim.LBFGS(self.NN_beta.parameters(), lr=0.1, line_search_fn='strong_wolfe')

    self.use_scheduler = use_scheduler

    if self.use_scheduler:
      self.sched_alpha_phase_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_NN_alpha_Adam_phase_1, mode='min', patience=1000)
      self.sched_beta_phase_1  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_NN_beta_Adam_phase_1, mode='min', patience=1000)

      self.sched_alpha_phase_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_NN_alpha_Adam_phase_2, mode='min', patience=1000)
      self.sched_beta_phase_2  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_NN_beta_Adam_phase_2, mode='min', patience=1000)

      self.sched_call  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_NN_call_Adam, mode='min', patience=1000)

  def call(self, x):
    # x = u, t, v
    return self.NN_call(x)

  def true_alpha(self, x):
    #result = torch.tensor(1.5) * (torch.tensor(0.5) - x[:, 2])
    result = x[:, 2] * (torch.tensor(1.0) - torch.tensor(10.0) * x[:, 2])
    return result

  def true_beta(self, x):
    #result = torch.tensor(0.7)
    result = torch.tensor(2.0) * x[: ,2]**(1.5)
    return result

  def loss_data(self):
    call_pred = self.call(self.dataset.x)
    call_true = self.dataset.price.reshape(-1, 1)
    with torch.no_grad():
      mean_price = torch.mean(torch.abs(call_true))
      weight_ = torch.clamp(mean_price / (torch.abs(call_true) + 1e-8), min=0.1, max=10)
    loss_data = torch.mean(weight_ * (call_pred - call_true) ** 2)
    return loss_data

  def loss_bound_fixed(self):
    t_zeros = torch.zeros((128,1), dtype = torch.float32).to(self.device)
    v_zeros = torch.zeros((128,1), dtype = torch.float32, requires_grad = True).to(self.device)
    u_rand = -self.u_min * torch.rand((128,1), dtype = torch.float32, device = self.device) + self.u_min
    t_rand = torch.rand((128,1), dtype = torch.float32, device = self.device)
    v_rand = (self.v_max - self.v_min) * torch.rand((128,1), dtype = torch.float32, device = self.device) + self.v_min

    t_rand.requires_grad = True

    x_rand_t0 = torch.cat((u_rand, t_zeros, v_rand), dim = 1)
    x_rand_v0 = torch.cat((u_rand, t_rand, v_zeros), dim = 1)

    call_true_t0 = torch.nn.functional.relu(1.0 - (self.K_max / self.S_0) * torch.exp(u_rand))
    call_true_v0 = torch.nn.functional.relu(1.0 - (self.K_max / self.S_0) * torch.exp(u_rand) * torch.exp(-self.r * self.T_max * t_rand), 0)

    call_pred_t0 = self.call(x_rand_t0)
    call_pred_v0 = self.call(x_rand_v0)

    with torch.no_grad():
        weight_t = torch.clamp(torch.mean(torch.abs(call_true_t0)) / torch.abs(call_true_t0), min=0.1, max=10)
        weight_v = torch.clamp(torch.mean(torch.abs(call_true_v0)) / torch.abs(call_true_v0), min=0.1, max=10)

    loss_t0 = torch.mean(weight_t * (call_pred_t0 - call_true_t0)**2)
    loss_v0 = torch.mean(weight_v * (call_pred_v0 - call_true_v0)**2)
    loss_bound = (loss_t0 + loss_v0)
    return loss_bound

  def loss_bound_flow(self):
    v_zeros = torch.zeros((128,1), dtype = torch.float32, requires_grad = True).to(self.device)
    u_rand = -self.u_min * torch.rand((128,1), dtype = torch.float32, device = self.device) + self.u_min
    t_rand = torch.rand((128,1), dtype = torch.float32, device = self.device)

    t_rand.requires_grad = True
    u_rand.requires_grad = True

    x_rand_v0 = torch.cat((u_rand, t_rand, v_zeros), dim = 1)

    call_pred_v0 = self.call(x_rand_v0)
    alpha_pred = self.NN_alpha(x_rand_v0)

    grad_call_v0_pred_t = torch.autograd.grad(call_pred_v0, t_rand, grad_outputs=torch.ones_like(call_pred_v0), create_graph=True)[0]
    grad_call_v0_pred_v = torch.autograd.grad(call_pred_v0, v_zeros, grad_outputs=torch.ones_like(call_pred_v0), create_graph=True)[0]

    with torch.no_grad():
      weight_dt = torch.clamp(torch.mean(torch.abs(grad_call_v0_pred_t)) / torch.abs(grad_call_v0_pred_t), min=0.1, max=10)

    v0_pde = (1 / self.T_max) * grad_call_v0_pred_t + alpha_pred * grad_call_v0_pred_v - self.r * call_pred_v0
    loss_v0_flow = torch.mean(weight_dt * (v0_pde)**2)
    return loss_v0_flow

  def loss_bound(self):
    return (1 / 3) * (self.loss_bound_fixed() + self.loss_bound_flow())

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

  def compute_grad_norm(self, loss, params):
        """Helper: Computes gradient norm for a specific loss w.r.t parameters."""
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, create_graph=False, allow_unused=True
        )
        grad_vec = torch.cat([g.view(-1) for g in grads if g is not None])
        if grad_vec.numel() == 0:
            return torch.tensor(0.0).to(self.device)
        return torch.norm(grad_vec)

  def get_adaptive_weights(self, losses, params):
        """
        Helper: Calculates the adaptive weights (lambdas) for a list of losses.
        Returns a list of detached tensors representing the weights.
        """
        norms = [self.compute_grad_norm(l, params) for l in losses]

        total_norm = sum(norms)

        weights = [torch.clamp((total_norm / (n + 1e-8)).detach(), max=100.0) for n in norms]

        return weights

  def initialize_adaptive_weights(self):
    loss_data = self.loss_data()
    loss_bound = self.loss_bound()
    loss_pde, loss_arb, loss_reg = self.loss_pde()
    losses = [loss_pde, loss_bound, loss_data, loss_arb]

    self.w_pde_alpha, self.w_bound_alpha = self.get_adaptive_weights(
                losses[:2], list(self.NN_alpha.parameters())
            )
    weights = self.get_adaptive_weights(losses, list(self.NN_call.parameters()))
    self.w_pde_call, self.w_bound_call, self.w_data_call, self.w_arb_call  = weights
    return self.w_pde_alpha, self.w_bound_alpha, self.w_data_call, self.w_arb_call, self.w_pde_call, self.w_bound_call

  def train_step_adam(self,update_weights = False):
    self.optimizer_NN_call_Adam.zero_grad()
    self.optimizer_NN_alpha_Adam_phase_1.zero_grad()
    self.optimizer_NN_beta_Adam_phase_1.zero_grad()

    # --- 1. Optimize Alpha (PDE + Bound) ---

    loss_bound = self.loss_bound()
    loss_pde, _, _ = self.loss_pde()

    if update_weights:
      w_pde_tilde, w_bound_tilde = self.get_adaptive_weights(
          [loss_pde, loss_bound], list(self.NN_alpha.parameters())
      )

      self.w_pde_alpha, self.w_bound_alpha = (0.9) * self.w_pde_alpha + 0.1 * w_pde_tilde, 0.9 * self.w_bound_alpha + 0.1 * w_bound_tilde

    if self.use_adaptive_loss_weights:
      loss_alpha = self.w_pde_alpha * loss_pde + self.w_bound_alpha * loss_bound
    else:
      loss_alpha = self.lambda_pde * loss_pde + self.lambda_bound * loss_bound
    loss_alpha.backward()

    for group in self.optimizer_NN_alpha_Adam_phase_1.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

    self.optimizer_NN_alpha_Adam_phase_1.step()

    self.optimizer_NN_call_Adam.zero_grad()
    self.optimizer_NN_alpha_Adam_phase_1.zero_grad()
    self.optimizer_NN_beta_Adam_phase_1.zero_grad()

    # --- 2. Optimize Beta (PDE Only) ---

    loss_pde, _, _ = self.loss_pde()
    loss_beta = loss_pde
    loss_beta.backward()

    for group in self.optimizer_NN_beta_Adam_phase_1.param_groups:
      torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

    self.optimizer_NN_beta_Adam_phase_1.step()

    self.optimizer_NN_call_Adam.zero_grad()
    self.optimizer_NN_alpha_Adam_phase_1.zero_grad()
    self.optimizer_NN_beta_Adam_phase_1.zero_grad()

    # --- 3. Optimize Call (Data + All others) ---

    loss_data = self.loss_data()
    loss_bound = self.loss_bound()
    loss_pde, loss_arb, loss_reg = self.loss_pde()

    if update_weights:
      weights_tilde = self.get_adaptive_weights([loss_pde, loss_bound, loss_data, loss_arb],
                                          list(self.NN_call.parameters())
                                          )
      self.w_data_call = 0.9 * self.w_data_call + 0.1 * weights_tilde[2]
      self.w_bound_call = 0.9 * self.w_bound_call + 0.1 * weights_tilde[1]
      self.w_pde_call = 0.9 * self.w_pde_call + 0.1 * weights_tilde[0]
      self.w_arb_call = 0.9 * self.w_arb_call + 0.1 * weights_tilde[3]

    if self.use_adaptive_loss_weights:
      loss_call = (
          self.w_data_call * loss_data +
          self.w_arb_call * loss_arb +
          self.w_pde_call * loss_pde +
          self.w_bound_call * loss_bound
      )
    else:
      loss_call = (
          self.lambda_data * loss_data +
          self.lambda_arb * loss_arb +
          self.lambda_pde * loss_pde +
          self.lambda_bound * loss_bound
          )
    loss_call.backward()

    for group in self.optimizer_NN_call_Adam.param_groups:
      torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

    self.optimizer_NN_call_Adam.step()

    if self.use_scheduler:
      self.sched_alpha_phase_1.step(loss_alpha.item())
      self.sched_beta_phase_1.step(loss_beta.item())
      self.sched_call.step(loss_call.item())

    return loss_data.item(), loss_bound.item(), loss_pde.item(), loss_arb.item(), loss_reg.item(), loss_call.item()

  def train_step_adam_alpha_beta_only(self,update_weights = False):

    if not hasattr(self, 'scaler_alpha_adam'):
      if self.use_adaptive_loss_weights:
        init_alpha = self.w_pde_alpha * self.pde_losses[-1] + self.w_bound_alpha * self.bound_losses[-1]
      else:
        init_alpha = self.lambda_pde * self.pde_losses[-1] + self.lambda_bound * self.bound_losses[-1]

      init_beta = self.pde_losses[-1]

      self.scaler_alpha_adam = 10.0 / (init_alpha + 1e-6)
      self.scaler_beta_adam = 10.0 / (init_beta + 1e-6)

    self.optimizer_NN_alpha_Adam_phase_2.zero_grad()
    self.optimizer_NN_beta_Adam_phase_2.zero_grad()

    # --- 1. Optimize Alpha (PDE + Bound) ---

    loss_bound = self.loss_bound_flow()
    loss_pde, _, _ = self.loss_pde()

    if update_weights:
      w_pde_tilde, w_bound_tilde = self.get_adaptive_weights(
          [loss_pde, loss_bound], list(self.NN_alpha.parameters())
      )

      self.w_pde_alpha, self.w_bound_alpha = (0.9) * self.w_pde_alpha + 0.1 * w_pde_tilde, 0.9 * self.w_bound_alpha + 0.1 * w_bound_tilde

    if self.use_adaptive_loss_weights:
      loss_alpha = self.w_pde_alpha * loss_pde + self.w_bound_alpha * loss_bound
    else:
      loss_alpha = self.lambda_pde * loss_pde + self.lambda_bound * loss_bound

    scaled_alpha = self.scaler_alpha_adam * loss_alpha
    scaled_alpha.backward()

    for group in self.optimizer_NN_alpha_Adam_phase_2.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

    self.optimizer_NN_alpha_Adam_phase_2.step()

    self.optimizer_NN_alpha_Adam_phase_2.zero_grad()
    self.optimizer_NN_beta_Adam_phase_2.zero_grad()

    # --- 2. Optimize Beta (PDE Only) ---

    loss_pde, _, _ = self.loss_pde()
    loss_beta = loss_pde
    scaled_beta = self.scaler_beta_adam * loss_beta
    scaled_beta.backward()

    for group in self.optimizer_NN_beta_Adam_phase_2.param_groups:
      torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

    self.optimizer_NN_beta_Adam_phase_2.step()

    self.optimizer_NN_alpha_Adam_phase_2.zero_grad()
    self.optimizer_NN_beta_Adam_phase_2.zero_grad()

    if self.use_scheduler:
      self.sched_alpha_phase_2.step(loss_alpha.item())
      self.sched_beta_phase_2.step(loss_beta.item())

    return loss_bound.item(), loss_pde.item()

  def train_step_LBFGS(self, skip_call):

    logs = {}

    if not hasattr(self, 'scaler_alpha'):
      if self.use_adaptive_loss_weights:
        init_alpha = self.w_pde_alpha * self.pde_losses[-1] + self.w_bound_alpha * self.bound_losses[-1]
      else:
        init_alpha = self.lambda_pde * self.pde_losses[-1] + self.lambda_bound * self.bound_losses[-1]

      init_beta = self.pde_losses[-1]

      if self.use_adaptive_loss_weights:
        init_call = (
            self.w_data_call * self.data_losses[-1] +
            self.w_arb_call * self.arb_losses[-1] +
            self.w_pde_call * self.pde_losses[-1] +
            self.w_bound_call * self.bound_losses[-1]
        )
      else:
        init_call = (
            self.lambda_data * self.data_losses[-1] +
            self.lambda_arb * self.arb_losses[-1] +
            self.lambda_pde * self.pde_losses[-1] +
            self.lambda_bound * self.bound_losses[-1]
            )

      self.scaler_alpha = 10.0 / (init_alpha + 1e-6)
      self.scaler_beta = 10.0 / (init_beta + 1e-6)
      self.scaler_call = 10.0 / (init_call + 1e-6)

    def closure_alpha():
      self.optimizer_NN_alpha_LBFGS.zero_grad()
      loss_bound = self.loss_bound()
      loss_pde, _, _ = self.loss_pde()
      if self.use_adaptive_loss_weights:
        loss_alpha = self.w_pde_alpha * loss_pde + self.w_bound_alpha * loss_bound
      else:
        loss_alpha = self.lambda_pde * loss_pde + self.lambda_bound * loss_bound
      scaled_alpha = self.scaler_alpha * loss_alpha
      scaled_alpha.backward()

      # --- Gradient Clipping ---
      for group in self.optimizer_NN_alpha_LBFGS.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

      logs['alpha_pde'] = loss_pde.item()
      logs['alpha_bound'] = loss_bound.item()
      logs['alpha_total'] = loss_alpha.item()

      return scaled_alpha

    def closure_beta():
      self.optimizer_NN_beta_LBFGS.zero_grad()
      loss_pde, _, _ = self.loss_pde()
      loss_beta = loss_pde
      scaled_beta = loss_beta * self.scaler_beta
      scaled_beta.backward()

      # --- Gradient Clipping ---
      for group in self.optimizer_NN_beta_LBFGS.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

      logs['beta_pde'] = loss_pde.item()
      logs['beta_total'] = loss_beta

      return scaled_beta

    def closure_call():
      self.optimizer_NN_call_LBFGS.zero_grad()
      loss_data = self.loss_data()
      loss_bound = self.loss_bound()
      loss_pde, loss_arb, loss_reg = self.loss_pde()
      if self.use_adaptive_loss_weights:
        loss_call = (
            self.w_data_call * loss_data +
            self.w_arb_call * loss_arb +
            self.w_pde_call * loss_pde +
            self.w_bound_call * loss_bound
        )
      else:
        loss_call = (
            self.lambda_data * loss_data +
            self.lambda_arb * loss_arb +
            self.lambda_pde * loss_pde +
            self.lambda_bound * loss_bound
            )

      scaled_call = loss_call * self.scaler_call
      scaled_call.backward()

      # --- Gradient Clipping ---
      for group in self.optimizer_NN_call_LBFGS.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1.0)

      logs['loss_call_data'] = loss_data.item()
      logs['loss_call_pde'] = loss_pde.item()
      logs['loss_call_bound'] = loss_bound.item()
      logs['loss_call_arb'] = loss_arb.item()
      logs['loss_call_reg'] = loss_reg.item()
      logs['loss_call_total'] = loss_call.item()

      return scaled_call

    loss_alpha = self.optimizer_NN_alpha_LBFGS.step(closure_alpha)
    loss_beta = self.optimizer_NN_beta_LBFGS.step(closure_beta)
    if not skip_call:
      loss_call = self.optimizer_NN_call_LBFGS.step(closure_call)

    return logs

  def run(self, single_phase = True):

    exact_alpha = self.true_alpha(self.dataset_unscaled.x)
    exact_beta = self.true_beta(self.dataset_unscaled.x)

    best_alpha_error = float('inf')
    best_beta_error = float('inf')
    best_call_loss = float('inf')
    best_epoch = 0
    model_alpha_save_path = "best_model_alpha.pth"
    model_beta_save_path = "best_model_beta.pth"
    model_call_save_path = "best_model_call.pth"

    self.data_losses = []
    self.bound_losses = []
    self.pde_losses = []
    self.arb_losses = []
    self.reg_losses = []
    self.call_losses = []
    self.alpha_errors = []
    self.beta_errors = []

    self.initialize_adaptive_weights()

    if single_phase:
      for epoch in range(self.num_epochs):
        if (epoch + 1) % 100 == 0:
          update_weights = True
        else:
          update_weights = False

        if epoch / self.num_epochs < 0.9 or not self.use_LBFGS:
            loss_data, loss_bound, loss_pde, loss_arb, loss_reg, loss_call = self.train_step_adam(update_weights)
        else:
          logs = self.train_step_LBFGS(skip_call = False)
          loss_data = logs['loss_call_data']
          loss_bound = logs['loss_call_bound']
          loss_pde = logs['loss_call_pde']
          loss_arb = logs['loss_call_arb']
          loss_call = logs['loss_call_total']

        if np.isnan(loss_call).any():
          print(f'Nan detected at epoch {epoch+1}')
          break

        self.data_losses.append(loss_data)
        self.bound_losses.append(loss_bound)
        self.pde_losses.append(loss_pde)
        self.arb_losses.append(loss_arb)
        self.reg_losses.append(loss_reg)
        self.call_losses.append(loss_call)

        alpha_pred = self.NN_alpha(self.dataset.x)
        beta_pred = self.NN_beta(self.dataset.x)

        alpha_error = torch.mean(torch.abs(alpha_pred - exact_alpha) / torch.abs(exact_alpha + 1e-8))
        beta_error = torch.mean(torch.abs(beta_pred - exact_beta) / torch.abs(exact_beta + 1e-8))
        alpha_beta_error = torch.maximum(alpha_error, beta_error)

        self.alpha_errors.append(alpha_error.item())
        self.beta_errors.append(beta_error.item())

        if (epoch+1) % 100 == 0:
          print(f'Epoch {epoch+1}')

          print(f'Data loss, {loss_data}')
          print(f'PDE loss, {loss_pde}')
          print(f'Arbitrage loss, {loss_arb}')
          print(f'Boundary loss, {loss_bound}')
          print(f'Alpha error, {alpha_error}')
          print(f'Beta error, {beta_error}')
          print('--------------------------------------')

        if loss_call < best_call_loss:
          best_call_loss = loss_call
          best_alpha_error = alpha_error
          best_beta_error = beta_error
          best_alpha_beta_error = alpha_beta_error
          best_epoch = epoch
          torch.save(self.NN_alpha.state_dict(), model_alpha_save_path)
          torch.save(self.NN_beta.state_dict(), model_beta_save_path)
          torch.save(self.NN_call.state_dict(), model_call_save_path)

    else:
      update_weights = False
      loss_data, loss_bound, loss_pde, loss_arb, loss_reg, loss_call = self.train_step_adam(update_weights)
      epoch_number = 1
      while loss_data > self.tol:
        if epoch_number > int(self.num_epochs / 2):
          print(f'Switched to phase two after {self.num_epochs / 2} epochs')
          break
        if epoch_number % 100 == 0:
          update_weights = True
        else:
          update_weights = False

        loss_data, loss_bound, loss_pde, loss_arb, loss_reg, loss_call = self.train_step_adam(update_weights)
        epoch_number += 1

        if np.isnan(loss_call).any():
          print(f'Nan detected at epoch {epoch_number}')
          break

        self.data_losses.append(loss_data)
        self.bound_losses.append(loss_bound)
        self.pde_losses.append(loss_pde)
        self.arb_losses.append(loss_arb)
        self.reg_losses.append(loss_reg)
        self.call_losses.append(loss_call)

        alpha_pred = self.NN_alpha(self.dataset.x)
        beta_pred = self.NN_beta(self.dataset.x)

        alpha_error = torch.mean(torch.abs(alpha_pred - exact_alpha) / torch.abs(exact_alpha + 1e-8))
        beta_error = torch.mean(torch.abs(beta_pred - exact_beta) / torch.abs(exact_beta + 1e-8))
        self.alpha_errors.append(alpha_error.item())
        self.beta_errors.append(beta_error.item())

        if (epoch_number) % 100 == 0:
          print(f'Epoch {epoch_number}')

          print(f'Data loss, {loss_data}')
          print(f'PDE loss, {loss_pde}')
          print(f'Arbitrage loss, {loss_arb}')
          print(f'Boundary loss, {loss_bound}')
          print(f'Alpha error, {alpha_error}')
          print(f'Beta error, {beta_error}')
          print('--------------------------------------')

        if loss_call < best_call_loss:
          best_call_loss = loss_call
          best_alpha_error = alpha_error
          best_beta_error = beta_error
          best_epoch = epoch_number
          torch.save(self.NN_alpha.state_dict(), model_alpha_save_path)
          torch.save(self.NN_beta.state_dict(), model_beta_save_path)
          torch.save(self.NN_call.state_dict(), model_call_save_path)

      print(f'Switched phase after {epoch_number} epochs')

      phase1_duration = epoch_number

      for epoch in range(int(self.num_epochs / 2)):
        current_epoch = phase1_duration + epoch
        if (epoch + 1) % 100 == 0:
          update_weights = True
        else:
          update_weights = False

        loss_flow, loss_pde = self.train_step_adam_alpha_beta_only(update_weights)

        loss_bound = self.loss_bound()

        if loss_bound.isnan().any() or np.isnan(loss_pde):
          print(f'Nan detected at epoch {epoch+1} in second phase')
          break

        self.bound_losses.append(loss_bound.item())
        self.pde_losses.append(loss_pde)
        self.data_losses.append(loss_data)
        self.arb_losses.append(loss_arb)

        if self.use_adaptive_loss_weights:
          loss_call = (
              self.w_data_call * loss_data +
              self.w_arb_call * loss_arb +
              self.w_pde_call * loss_pde +
              self.w_bound_call * loss_bound.item()
          )
        else:
          loss_call = (
              self.lambda_data * loss_data +
              self.lambda_arb * loss_arb +
              self.lambda_pde * loss_pde +
              self.lambda_bound * loss_bound.item()
              )

        self.call_losses.append(loss_call.item())


        alpha_pred = self.NN_alpha(self.dataset.x)
        beta_pred = self.NN_beta(self.dataset.x)

        alpha_error = torch.mean(torch.abs(alpha_pred - exact_alpha) / torch.abs(exact_alpha))
        beta_error = torch.mean(torch.abs(beta_pred - exact_beta) / exact_beta)
        self.alpha_errors.append(alpha_error.item())
        self.beta_errors.append(beta_error.item())

        if (epoch+1) % 1000 == 0:
          print(f'Epoch {epoch+1}')

          print(f'PDE loss, {loss_pde}')
          print(f'Boundary loss, {loss_bound.item()}')
          print(f'Alpha error, {alpha_error}')
          print(f'Beta error, {beta_error}')
          print('--------------------------------------')

        if loss_call < best_call_loss:
          best_call_loss = loss_call
          best_alpha_error = alpha_error
          best_beta_error = beta_error
          best_epoch = current_epoch
          torch.save(self.NN_alpha.state_dict(), model_alpha_save_path)
          torch.save(self.NN_beta.state_dict(), model_beta_save_path)
          torch.save(self.NN_call.state_dict(), model_call_save_path)


    print(f'Best alpha error {best_alpha_error}')
    print(f'Best beta error {best_beta_error}')
    print(f'Best call loss {best_call_loss}')
    print(f'Best model paths loaded from epoch {best_epoch}')
    self.NN_alpha.load_state_dict(torch.load(model_alpha_save_path))
    self.NN_beta.load_state_dict(torch.load(model_beta_save_path))
    self.NN_call.load_state_dict(torch.load(model_call_save_path))

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
    plt.plot(t_all.cpu().numpy(), S_path_true_np, lw=1, label='SDE with exact alpha and beta', linestyle='--')
    plt.plot(t_all.cpu().numpy(), S_path_pred_np, lw=1, label='SDE with Modeled alpha and beta')
    plt.title('Asset Price Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    self.asset_trajectory_plot = fig
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=[12, 3], dpi=450)
    plt.plot(t_all.cpu().numpy(), v_path_true_np, lw=1, label='SDE with exact alpha and beta', linestyle='--')
    plt.plot(t_all.cpu().numpy(), v_path_pred_np, lw=1, label='SDE with Modeled alpha and beta')
    plt.title('Volatility Trajectories')
    plt.xlabel('Time')
    plt.ylabel('Volatility Price')
    self.vol_trajectory_plot = fig
    plt.show()
    plt.close()

  def plot_epochs(self):
    losses = {
        'Data Loss': self.data_losses,
        'Boundary Loss': self.bound_losses,
        'PDE Loss': self.pde_losses,
        'Arb Loss': self.arb_losses,
        'Call Loss': self.call_losses
    }
    num_plots = len(losses)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 12), sharex=True)

    epochs_range = list(range(1, len(self.data_losses) + 1))


    for ax, (loss_name, loss_values) in zip(axes.flatten(), losses.items()):

        ax.plot(epochs_range, loss_values, label=loss_name, color='tab:blue', linewidth=1.5)

        # Optional: Use log scale if your losses span different orders of magnitude (common in PDEs)
        ax.set_yscale('log')

        ax.set_title(loss_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelbottom=True)

    axes[-1].set_xlabel('Epochs / Iterations')
    self.epochs_plot = fig
    plt.tight_layout()
    plt.show()

  def plot_alpha_beta(self):
    errors = {
        'Alpha errors': self.alpha_errors,
        'Beta errors': self.beta_errors
    }
    num_plots = len(errors)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 12), sharex=True)

    epochs_range = list(range(1, len(self.alpha_errors) + 1))

    for ax, (error_name, error_vals) in zip(axes.flatten(), errors.items()):

        ax.plot(epochs_range, error_vals, label=error_name, color='tab:blue', linewidth=1.5)

        # Optional: Use log scale if your losses span different orders of magnitude (common in PDEs)
        #ax.set_yscale('log')

        ax.set_title(error_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Value')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelbottom=True)

    axes[-1].set_xlabel('Epochs / Iterations')
    self.alpha_beta_error_plot = fig
    plt.tight_layout()
    plt.show()

  def save_plots(self):
    self.asset_trajectory_plot.savefig('asset_trajectory_plot.png')
    self.vol_trajectory_plot.savefig('vol_trajectory_plot.png')
    self.epochs_plot.savefig('epochs_plot.png')
