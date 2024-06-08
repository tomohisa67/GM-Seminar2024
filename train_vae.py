import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
from models.vae import VAE, loss_function
from data.dataset import get_dataloader
from utils.utils import save_checkpoint, save_model
import json
import os
import argparse
from typing import Dict, Any

def train_vae(epoch: int, model: VAE, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    train_loss = 0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % 10 == 0:
            print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test_vae(epoch: int, model: VAE, test_loader: DataLoader, device: torch.device) -> None:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use (default: mnist)')
    parser.add_argument('--config', type=str, default='configs/config_vae.json', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config: Dict[str, Any] = json.load(f)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloader(dataset_name=args.dataset, batch_size=config["batch_size"])
    
    model: VAE = VAE(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"]).to(device)
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    save_dir: str = 'save'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path: str = os.path.join(save_dir, 'model.pth')

    # training
    for epoch in range(1, config["epochs"] + 1):
        train_vae(epoch, model, train_loader, optimizer, device)
        test_vae(epoch, model, test_loader, device)
        save_checkpoint({
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }, filename=config["save_path"])

    # save model
    save_model(model, model_save_path)

if __name__ == '__main__':
    main()
