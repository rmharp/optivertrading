import torch
import torch.nn as nn

class OptiverNet(torch.nn.Module):
    def __init__(self, num_layers=20, hidden_size=256):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(12, hidden_size))
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_size, 1))
        # for i in range(num_layers):
        #     self.layers.append(torch.nn.Linear(12, 12))
        #     self.layers.append(torch.nn.ReLU())
        # self.layers.append(torch.nn.Linear(12, 1))
        # self.linear1= torch.nn.Linear(12, 256)
        # self.relu1 = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(256, 128)
        # self.relu2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(128, 64)
        # self.relu3 = torch.nn.ReLU()
        # self.linear4 = torch.nn.Linear(64, 32)
        # self.relu4 = torch.nn.ReLU()
        # self.linear5 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        # x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        # x = self.linear3(x)
        # x = self.relu3(x)
        # x = self.linear4(x)
        # x = self.relu4(x)
        # x = self.linear5(x)
        # return x