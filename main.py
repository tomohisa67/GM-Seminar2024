import torch
from models.vae import VAE
from data.dataset import get_dataloader
from utils.utils import load_model
import json
import argparse
import os
import matplotlib.pyplot as plt

def infer_vae(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
    return recon_batch

def main():
    parser = argparse.ArgumentParser(description='Inference with VAE')
    parser.add_argument('--model_type', type=str, required=True, help='Model type to use (vae)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use (default: mnist)')
    parser.add_argument('--config', type=str, default='configs/config_vae.json', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloader(dataset_name=args.dataset, batch_size=config["batch_size"])
    
    if args.model_type == 'vae':
        model = VAE(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"])
        model = load_model(model, args.checkpoint, device)

        test_data_iter = iter(test_loader)
        test_data, _ = next(test_data_iter)
        reconstructed = infer_vae(model, test_data, device)
        
        # plot
        plt.figure(figsize=(9, 2))
        for i in range(9):
            # original images
            plt.subplot(2, 9, i+1)
            plt.imshow(test_data[i].cpu().numpy().reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title('Original')
            # reconstructed images
            plt.subplot(2, 9, i+10)
            plt.imshow(reconstructed[i].cpu().numpy().reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title('Reconstructed')
        plt.show()

if __name__ == '__main__':
    main()
