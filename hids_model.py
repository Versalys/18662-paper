# Reasoning Based AI for Malware Detection Through State Abstraction
# Written by Aidan Erickson
#
# This code includes the LSTM based model for the paper above. Additional information
# can be found in main.py.


from torch import nn
import torch
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU acceleration
# TODO: Look at hidden states. Update lstm to not pass tuple. Use DL homework.

class HidsModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, embed_dim, num_layers=5):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, device=DEVICE)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, device=DEVICE)

        self.linear = nn.Linear(hidden_dim, output_dim, device=DEVICE)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.linear(x)[:, -1]
        x = self.sigmoid(x)
        return x
