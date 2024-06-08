import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, code_dim: int = 64) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 784)  # フラット化
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        return reconstructed

def autoencoder_loss_function(recon_x: Tensor, x: Tensor) -> Tensor:
    return F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

