import os
import numpy as np
import torch

def save_checkpoint(state, filename="logs/checkpoint.pth.tar"):
    torch.save(state, filename)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_output(data, save_path, file_name='reconstructed.npy'):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, file_name)
    np.save(save_path, data)
    print(f'Model output is saved as numpy array to {save_path}')

