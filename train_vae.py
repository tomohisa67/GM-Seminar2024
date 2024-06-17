import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.vae import VAE, ConvVAE
from data.dataset import get_dataloader
from utils.utils import save_checkpoint, save_model
import json
import os
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt

def loss_function(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, reg: float) -> Tensor:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + reg * KLD, BCE, KLD

def loss_function2(recon_x, x, mu, logvar, reg): # for ConvVAE
    recon_x = recon_x.view(recon_x.size(0), -1)  # Flatten recon_x to (batch_size, input_dim)
    x = x.view(x.size(0), -1)  # Flatten x to (batch_size, input_dim)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + reg * KLD, BCE, KLD

def train_vae(epoch: int, model: VAE, reg: float, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    train_loss = 0
    recon_loss_total = 0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, reg)
        loss.backward()
        train_loss += loss.item()
        recon_loss_total += recon_loss.item()
        optimizer.step()
        if i % 10 == 0:
            print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    average_recon_loss = recon_loss_total / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}, Reconstruction loss: {average_recon_loss:.4f}')
    return average_recon_loss

def test_vae(epoch: int, model: VAE, reg: float, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    test_loss = 0
    recon_loss_total = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar, reg)
            test_loss += loss.item()
            recon_loss_total += recon_loss.item()
    test_loss /= len(test_loader.dataset)
    average_recon_loss = recon_loss_total / len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}, Reconstruction loss: {average_recon_loss:.4f}')
    return average_recon_loss

def main() -> None:
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use (default: mnist)')
    parser.add_argument('--config', type=str, default='configs/config_vae.json', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config: Dict[str, Any] = json.load(f)

    dataset_name = args.dataset
    batch_size = config["batch_size"]
    input_dim = config["input_dim"]
    input_channels = config["input_channels"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    reg = config["reg"]
    lr = config["learning_rate"]
    epochs = config["epochs"]
    save_log_path = config["save_path"]

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size)
    
    model: VAE = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    # model: ConvVAE = ConvVAE(input_channels=input_channels, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=lr)

    save_dir: str = 'save'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path: str = os.path.join(save_dir, 'model.pth')

    train_recon_losses = []
    test_recon_losses = []

    # training
    for epoch in range(1, epochs + 1):
        train_recon_loss = train_vae(epoch, model, reg, train_loader, optimizer, device)
        test_recon_loss = test_vae(epoch, model, reg, test_loader, device)
        train_recon_losses.append(train_recon_loss)
        test_recon_losses.append(test_recon_loss)
        save_checkpoint({
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }, filename=save_log_path)

    # save model
    save_model(model, model_save_path)

    # plot reconstruction loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_recon_losses, marker='o', label='Train Reconstruction Loss')
    plt.plot(range(1, epochs + 1), test_recon_losses, marker='o', label='Test Reconstruction Loss')
    plt.title('Reconstruction Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    # plt.savefig(os.path.join(save_dir, 'reconstruction_loss.png'))
    plt.show()

if __name__ == '__main__':
    main()
