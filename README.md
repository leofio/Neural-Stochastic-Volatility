# Neural-Stochastic-Volatility
A neural network implementation for calibrating stochastic volatility models using PyTorch. Includes a custom class for simulating synthetic price paths and option Greeks. 

## Files overview
* **synthetic_data.py** : Implements Monte Carlo simulation to generate synthetic option price data for model testing
* **datasets.py** : Datasets class for model training
* **nets.py** : Defines network architecure for learning dynamics
* **trainer.py** : Handles training logic. Manages optimization and loss tracking

