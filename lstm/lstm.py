import torch.nn as nn
import os
 
class OptiverModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = int(os.environ['HIDDEN_SIZE'])
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=int(os.environ['LSTM_LAYERS']), batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
    def forward(self, x):
        x, _ = self.lstm(x)
        print(_[0].shape)
        x = self.linear(x)
        return x