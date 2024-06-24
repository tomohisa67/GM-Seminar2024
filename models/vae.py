import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def conv_output_size(h, kernel_size=4, stride=2, padding=1):
    return (h + 2 * padding - kernel_size) // stride + 1

def convtranspose_output_size(h, kernel_size=4, stride=2, padding=1):
    return (h - 1) * stride - 2 * padding + kernel_size

def conv_output_size(h, kernel_size=4, stride=2, padding=1):
    return (h + 2 * padding - kernel_size) // stride + 1

def convtranspose_output_size(h, kernel_size=4, stride=2, padding=1):
    return (h - 1) * stride - 2 * padding + kernel_size

class ConvVAE(nn.Module):
    def __init__(self, input_channels: int = 3, hidden_dim: int = 128, latent_dim: int = 20, input_size: int = 32) -> None:
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: [32, input_size//2, input_size//2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, input_size//4, input_size//4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [128, input_size//8, input_size//8]
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=4, stride=1, padding=0),  # Output: [hidden_dim, (input_size//8)-3, (input_size//8)-3]
            nn.ReLU()
        )

        # Calculate the flattened size after the encoder
        self.flatten = nn.Flatten()
        self.encoder_output_size = self._get_encoder_output_size()

        # Fully connected layers for mean and log variance
        self.fc1 = nn.Linear(self.encoder_output_size, self.latent_dim)  # mean
        self.fc2 = nn.Linear(self.encoder_output_size, self.latent_dim)  # logvar
        self.fc3 = nn.Linear(self.latent_dim, self.encoder_output_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=1, padding=0),  # Output: [128, (input_size//8)-3, (input_size//8)-3]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, input_size//4, input_size//4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: [32, input_size//2, input_size//2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: [input_channels, input_size, input_size]
            nn.Sigmoid()
        )

    def _get_encoder_output_size(self):
        size = self.input_size
        size = conv_output_size(size)
        size = conv_output_size(size)
        size = conv_output_size(size)
        size = conv_output_size(size, stride=1, padding=0)
        return size * size * self.hidden_dim

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = self.flatten(h)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc3(z)
        h = h.view(h.size(0), self.hidden_dim, (self.input_size // 8) - 3, (self.input_size // 8) - 3)
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar