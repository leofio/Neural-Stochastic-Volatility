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

The overal loss for the PINN 
