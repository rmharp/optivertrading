import torch.nn as nn
import os
 
class OptiverModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=50, num_layers=int(os.environ['LSTM_LAYERS']), batch_first=True)
        self.linear = nn.Linear(in_features=50, out_features=1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x