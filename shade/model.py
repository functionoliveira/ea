import torch
import torch.nn as nn

n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01

data_x = torch.randn(batch_size, n_input)
data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()


class DummyNeuralNet(nn.Module):
    def __init__(self):
        super(DummyNeuralNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)
