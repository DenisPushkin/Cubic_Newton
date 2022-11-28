import torch.nn as nn


class LogisticNet(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    
    def __init__(self, dims, activation, output_activation=None):
        super().__init__()
        assert len(dims) >= 2
        self.activation = activation
        self.output_activation = output_activation
        self.n_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
    
    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[self.n_layers - 1](x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x