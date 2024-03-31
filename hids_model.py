from torch import nn
import torch
from torch import nn.functional as F


class HidsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.modules = nn.Sequential(nn.LSTM(input_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LeakyReLU()
                                     nn.Linear(hidden_dim, output_dim)
                                     )

    def forward(x):
        x = self.modules(x)
        return x
