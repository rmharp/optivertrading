import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
class OptiverModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=50, num_layers=4, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x