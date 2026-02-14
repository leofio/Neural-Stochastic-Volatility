import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class NN_Call(nn.Module):
    def __init__(self, nn_params):
        super(NN_Call, self).__init__()
        [input_size, output_size, hidden_layers, num_neurons, neurons_per_layer, activation, use_batch_norm, dropout_rate] = nn_params
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
      return F.softplus(self.layers(x))

class NetAlpha(nn.Module):
    def __init__(self, nn_params):
        super(NetAlpha, self).__init__()
        [input_size, output_size, hidden_layers, num_neurons, neurons_per_layer, activation, use_batch_norm, dropout_rate] = nn_params
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

class NetBeta(nn.Module):
    def __init__(self, nn_params):
        super(NetBeta, self).__init__()
        [input_size, output_size, hidden_layers, num_neurons, neurons_per_layer, activation, use_batch_norm, dropout_rate] = nn_params
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
