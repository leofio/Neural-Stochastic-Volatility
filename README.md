# Neural-Stochastic-Volatility
A neural network implementation for calibrating stochastic volatility models using PyTorch. Includes a custom class for simulating synthetic price paths and option Greeks. 

## Files overview
* **synthetic_data.py** : Implements Monte Carlo simulation to generate synthetic option price data for model testing
* **datasets.py** : Datasets class for model training
* **nets.py** : Defines network architecure for learning dynamics
* **trainer.py** : Handles training logic. Manages optimization and loss tracking. Includes four different training procedures.
  * **Single Phase** : The three networks are optimized in turn with a coordinate decent algorithm, first with Adam then with LBFGS
  * **Single Phase Joint** : The three networks are optimized jointly with respect to the total loss, first with Adam then with LBFGS
  * **Dual Phase type I** : In the first phase all three networks are optimized with coordinate descent then, once the data loss falls below a set tolerance, the solution surface is held fixed and only \alpha_{\nu,t} dt and \beta_{\nu,t} are optimized
  * **Dual Phase type II** : In the fisrt phase only the call suface is fit with respect to just the data loss then it is held fixed and \alpha_{\nu,t} dt and \beta_{\nu,t} are optimized. This is not good practice but is included to show the importance of the PDE loss.

## Project overview
Inspired by the work of Wang et al. [2025] on Deep self-consistent learning of local volatility, this repository extends that approach to Stochastic Volatility (SV) models. This requires handling additional latent parameters and more complex state dynamics in the synthetic data generation phase. Similarly, I used physics informed neural networks (PINNs) to learn a consistent pricing surface and the underlying asset and volatility dynamics.

### Mathematical setup
Asssuing a stochstic volatility model for the asset price and volatility, the goal is then to recover the unknown dynamics $\alpha_{\nu,t}$ and $\beta_{\nu,t}$.

$$
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{\nu_t} S_t dW_t \\
d\nu_t &= \alpha_{\nu,t} dt + \beta_{\nu,t} dB_t
\end{aligned}
$$

I analyse the corresponding PDE of the pricing surface to learn $\alpha_{\nu,t}$, $\beta_{\nu,t}$, and the pricing surface $V(S_t,t,v)$ using PINNs.

## References
- **Wang, et al.** - *[Deep self-consistent learning of local volatility]* ([2025]). [(https://arxiv.org/abs/2201.07880)]
  - This project adapts the neural network calibration architecture proposed by Wang et al. for local volatility, extending the methodology to **Stochastic Volatility** dynamics.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details
