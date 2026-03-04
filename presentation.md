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
We have $C( \lambda S, \lambda K) = \lambda C(S,K)$ so that, by Euler's Homogeneous Function Theorem $C = S \frac{\partial C}{\partial S} + K \frac{\partial C}{\partial K}$. Applying this change of variables give the following PDE.

$$
\frac{\partial C}{\partial \tau} = -rK \frac{\partial C}{\partial K} + (\alpha_{\nu,t} + \rho \beta_{\nu,t} \nu) \frac{\partial C}{\partial \nu} + \frac{1}{2} \nu K^2 \frac{\partial^2 C}{\partial K^2} + \frac{1}{2}\beta_{\nu,t} ^ 2 \nu \frac{\partial^2 C}{\partial \nu^2} - \rho \beta_{\nu,t} \nu K \frac{\partial^2 C}{\partial K \partial V}
$$
