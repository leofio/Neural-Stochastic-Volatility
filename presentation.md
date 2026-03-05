---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  h1 {
    color: #2c3e50;
  }
---

# Deep self-consistent learning of stochastic volatility models
## Dissertation Presentation

**Leonardo Fiore**
Mathematics Department
Unviersity of Oxford

---

## Stochastic Volatility Models

Stochastic volatilty models describe how an assets price evolves over time using a system of coupled stochastic differential equations.


$$
\begin{aligned}
dS_t &= r S_t dt + \sqrt{\nu_t} S_t dW_t \\
d\nu_t &= \alpha_{\nu,t} dt + \beta_{\nu,t} \sqrt{\nu_t} dB_t
\end{aligned}
$$

&nbsp;


$$
\langle dW_t^S, dW_t^\nu \rangle = \rho dt
$$

The goal of this project is the recover function(s) $\alpha_{\nu,t}$ and $\beta_{\nu,t}$ from a dataset of options prices derived from an asset that follows this model.
$r$ is the risk free rate and these equations are taken in the risk neutral measure space.

---

# Options pricing
## European Call options

* A European call option is a derivative contract that gives the holder the right (but not the obligation) to purchase an underlying asset at a pre-specified price (strike) on a specific expiration date

* For a European call option derived from asset $S_t$, with strike K and maturity T the payoff is given by

$$
\Phi(S_T) = (S_T - K)^+ = \max(S_T - K, 0)
$$

* The option price at time $t$ is given by the following formula where $\mathcal{F}_t$ contains the information of both $S_t$ and $\nu_t$ up to time $t$

$$
C_t = e^{-r(T-t)} \mathbb{E}^\mathbb{Q} [ \max(S_T - K, 0) \mid \mathcal{F}_t ]
$$

---

# Backward Kolmogorov Equation

The BKE states that $u(t,S,V) = \mathbb{E} [\Phi(S_T)]$ is goverend by the followign PDE where $\mathcal{L}$ is the infinitesimal generator of the process

$$
\frac{\partial u}{\partial t} + \mathcal{L}u = 0
$$

## Infinitesimal Generator $\mathcal{L}$

For the stochastic volatility model, the generator $\mathcal{L}$ is given by:

$$
\mathcal{L} = rS \frac{\partial}{\partial S} + \mu \frac{\partial}{\partial V} + \frac{1}{2}VS^2 \frac{\partial^2}{\partial S^2} + \frac{1}{2}\sigma^2 \frac{\partial^2}{\partial V^2} + \rho \sqrt{V} S \sigma \frac{\partial^2}{\partial S \partial V}
$$

---

# Call option price surface

Option price is the discounted expected payoff $C_t = e^{-r(T-t)} \mathbb{E}^\mathbb{Q} [ \max(S_T - K, 0)]$
In the BKE this becomes

$$
\frac{\partial C}{\partial t} + \mathcal{L}C - rC = 0
$$

$$
\frac{\partial C}{\partial t} + rS \frac{\partial C}{\partial S} + \alpha_{\nu,t} \frac{\partial C}{\partial V} + \frac{1}{2}VS^2 \frac{\partial^2 C}{\partial S^2} + \frac{1}{2}\beta_{\nu,t} ^ 2 V \frac{\partial^2 C}{\partial V^2} + \rho \beta_{\nu,t} V S \frac{\partial^2 C}{\partial S \partial V} - rC = 0
$$

Let $\tau = T-t$ be the time to maturity and $K$ the strike price. 
We have $C( \lambda S, \lambda K) = \lambda C(S,K)$ so that, by Euler's Homogeneous Function Theorem, $C = S \frac{\partial C}{\partial S} + K \frac{\partial C}{\partial K}$. Applying this change of variables gives the following PDE.

$$
\frac{\partial C}{\partial \tau} = -rK \frac{\partial C}{\partial K} + (\alpha_{\nu,\tau} + \rho \beta_{\nu,\tau} \nu) \frac{\partial C}{\partial \nu} + \frac{1}{2} \nu K^2 \frac{\partial^2 C}{\partial K^2} + \frac{1}{2}\beta_{\nu,\tau} ^ 2 \nu \frac{\partial^2 C}{\partial \nu^2} - \rho \beta_{\nu, \tau} \nu K \frac{\partial^2 C}{\partial K \partial V}
$$

---

# Boundary Conditions

The PDE has the following boundary conditions in the transformed variables.

### At $\nu_\tau = 0$

$-\frac{\partial C}{\partial \tau} + \alpha_{\nu = 0, \tau} C_\nu = 0$

&nbsp;

### At $\tau = 0$

$C = \max(S_0 - K, 0)$

---

# Arbitrage free surface

Real world finiancial markets must be arbitrage free, this can be enforced by the following conditions.

$$
\frac{\partial C}{\partial \tau} \geq 0
$$

$$
\frac{\partial ^2 C}{\partial K^2} \geq 0
$$

---

<style scoped>
section {
  font-size: 24px;
}
</style>

# Theoretical Identifiabilty 

$\alpha_{t, \nu}$ and $\beta_{t, \nu}$ are taken to be independent of $K$ because this provides theoretical uniqueness of $\alpha_{t, \nu}$, given a solution $C$.

### Proof

Write

$$
\small
F(\tau, K, \nu) = -\frac{\partial C}{\partial \tau} -rK \frac{\partial C}{\partial K} + \frac{1}{2} \nu K^2 \frac{\partial^2 C}{\partial K^2}
$$

then

$$
\small
 F(\tau, K ,\nu) + (\alpha_{\nu,\tau} + \rho \beta_{\nu,\tau} \nu) \frac{\partial C}{\partial \nu} + \frac{1}{2}\beta_{\nu,\tau} ^ 2 \nu \frac{\partial^2 C}{\partial \nu^2} - \rho \beta_{\nu, \tau} \nu K \frac{\partial^2 C}{\partial K \partial V} = 0\\
 $$

and, by independece from K,

$$
\small
 \frac{\partial F}{\partial K} + (\alpha_{\nu,\tau} + \rho \beta_{\nu,\tau} \nu) \frac{\partial ^2 C}{\partial \nu \partial K} + \frac{1}{2}\beta_{\nu,\tau} ^ 2 \nu \frac{\partial^3 C}{\partial \nu^2 \partial K} - \rho \beta_{\nu, \tau} \nu K \frac{\partial^3 C}{\partial K^2 \partial V} = 0
$$

This gives two equations for $\alpha_{t, \nu}$ and $\beta_{t, \nu}$ that, with the convention $\beta_{t, \nu} \geq 0$, determine $\alpha_{t, \nu}$ and $\beta_{t, \nu}$.


---

# Inverse PINNs

Inverse PINNs are used in scientific deep learning to recover unkown parameters or functions from a dynamical sysmtem. The inverse PINN simultaneously reconstructs the full solution field and discovers those unknown physical parameters.

## Procedure

$C$, $\alpha_{\nu,\tau}$ and $\beta_{\nu,\tau}$ are all modeled with neural networks. $C$ fits the data and PDE loss, then $\alpha_{\nu,\tau}$ and $\beta_{\nu,\tau}$ are recovered via the PDE loss.

---

# Scaling

Scaling varibales is improtant for two reasons;
  * It can help simplify the PDE
  * The neural networks should take scaled inputs for improved results

$$
t = \frac{\tau}{\tau_{max}}, u = \ln \left( \frac {e^{-r\tau} K}{K_{max}} \right)
$$

The PDE becomes

$$
\frac{1}{\tau_{max}} \frac{\partial C}{\partial t} = \frac{1}{2} \nu (\frac{\partial^2 C}{\partial u^2} - \frac{\partial C}{\partial u}) - \rho \beta \nu \frac{\partial^2 C}{\partial u \partial v} + \frac{1}{2} \beta^2 \nu \frac{\partial^2 C}{\partial v^2} + (\alpha + \rho \beta \nu) \frac{\partial C}{\partial \nu}
$$

---

# Networks

## Call option surface

$$
C(u, t, \nu) = S_0 \mathcal{N}_{call}(u, t, \nu)
$$

4 hidden layers each with 64 neurons

## Alpha and Beta

$$
\alpha_{t, \nu} = \mathcal{N}_{alpha}(t, \nu)
$$

$$
\beta_{t, \nu} = \mathcal{N}_{beta}(t, \nu)
$$

2 hidden layers each with 10 neurons

---

# Losses

The overal loss for the PINN is split into four terms that each enforce a desired behaviour.

$$
\mathcal{L}_{total} = \lambda_{data}\mathcal{L}_{data} + \lambda_{pde}\mathcal{L}_{pde} + \lambda_{bound}\mathcal{L}_{bound} + \lambda_{arb}\mathcal{L}_{arb} 
$$

$$
\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} (S_0 \mathcal{N}_{call}(u_i, t_i, \nu_i) - C_i)^2
\quad
\mathcal{L}_{pde} = \frac{1}{M_1} \sum_{i=1}^{M_1} f_{pde}(u_i, t_i, \nu_i)
$$

$$
\mathcal{L}_{bound} = \frac{1}{M_2} \sum_{i=1}^{M_2} f_{bound}(u_i, t_i, \nu_i)
\quad
\mathcal{L}_{arb} = \frac{1}{M_3} \sum_{i=1}^{M_3} f_{arb}(u_i, t_i, \nu_i)
$$

Where $f_{pde}, f_{bound}$ are the residuals we need to minimize, $f_{arb}$ penalizes the arbitrage conditionds, $M_1, M_2, M_3$ are the number of collocation points sampled, $\lambda_{data}, \lambda_{pde}, \lambda_{bound}, \lambda_{arb}$ are scaling weights.

---

# Loss weighting

The choice of $\lambda_i$ is important in the training of the PINN. I implemented an adaptive loss weighting system to improve training.

$$
\hat{\lambda}_{i} = \frac{\sum_{j=1}^{4}\|\nabla_{\theta}\mathcal{L}_{j}(\theta)\|}{\|\nabla_{\theta}\mathcal{L}_{i}(\theta)\| + \epsilon},
$$

$$
\lambda_{new} = \alpha \lambda_{old} + (1-\alpha) \hat{\lambda}_{new}
$$

Weights are updataed every 100 epochs with $\alpha = 0.9$. Weights are clamped to $100$ and $\epsilon = 1e-8$ to stop the weights exploding. 

---

# Data generation

I generated synthetic option price data using Monte Carlo simulation with known models.

#### Heston model
$$
\small
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\
dv_t &= \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} dW_t^v
\end{aligned}
$$

#### 3/2 model
$$
\small
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\
dv_t &= \kappa v_t (\theta - v_t) dt + \sigma v_t^{3/2} dW_t^v
\end{aligned}
$$

![w:500](images/asset_trajectories.png) ![w:500](images/volatility_trajectories.png)

---

# Training

I tested four different training strategies.

1. **Single Phase** : The three networks are optimized in turn with a coordinate decent algorithm, first with Adam then with LBFGS
2. **Single Phase Joint** : The three networks are optimized jointly with respect to the total loss, first with Adam then with LBFGS
3. **Dual Phase type I** : In the first phase all three networks are optimized with coordinate descent then, once the data loss falls below a set tolerance, the solution surface is held fixed and only $\alpha_{\nu,t}$ and $\beta_{\nu,t}$ are optimized
4. **Dual Phase type II** : In the fisrt phase only the call suface is fit with respect to just the data loss then it is held fixed and $\alpha_{\nu,t}$ and $\beta_{\nu,t}$ are optimized. This is not good practice but is included to show the importance of the PDE loss.

---

# Heston Results - beta fixed

![h:500 w:](images/epochs_plot_heston_new.png)

![bg right vertical h:300](images/alpha_plot_heston_new.png)
![bg right vertical h:300](images/call_price_plot_heston_new.png)

---

# Heston Results alpha and beta unknown

![h:500 w:](images/call_price_plot_fixed_strike_mixed_heston.png)

![bg right vertical h:300](images/alpha_surface_plot_mixed_heston.png)
![bg right vertical h:300](images/beta_plot_mixed_heston.png)

---

# 3/2 model results

![h:500 w:](images/call_price_plot_fixed_strike_three_two.png)

![bg right vertical h:300](images/alpha_surface_plot_three_two.png)
![bg right vertical h:300](images/beta_surface_plot_three_two.png)

---

# References

- **Wang, Z., Shaa, A., Privault, N., & Guet, C. (2021).** *Deep self-consistent learning of local volatility.* arXiv preprint arXiv:2201.07880. [https://doi.org/10.48550/arXiv.2201.07880](https://doi.org/10.48550/arXiv.2201.07880)

- **Gatheral, J. (2006).** *The Volatility Surface: A Practitioner's Guide.* John Wiley & Sons. ISBN: 978-0-471-79251-2.

- **Zhou, W., & Xu, Y. F. (2021).** *Data-Guided Physics-Informed Neural Networks for Solving Inverse Problems in Partial Differential Equations.* arXiv preprint arXiv:2104.05386. [https://doi.org/10.48550/arXiv.2104.05386](https://doi.org/10.48550/arXiv.2104.05386)

- **Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023).** *An Expert's Guide to Training Physics-Informed Neural Networks.* arXiv preprint arXiv:2308.08468. [https://doi.org/10.48550/arXiv.2308.08468](https://doi.org/10.48550/arXiv.2308.08468)
