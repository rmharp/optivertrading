import torch
import torch.nn as nn

class OptiverNet(torch.nn.Module):
    def __init__(self, num_layers=10, hidden_size=256):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(12, hidden_size))
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(0.3))
        self.layers.append(torch.nn.Linear(hidden_size, 1))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x