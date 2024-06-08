import torch
from torch import nn
import os
import numpy as np
from typing import Any, Dict

def save_checkpoint(state: Dict[str, Any], filename: str = "logs/checkpoint.pth.tar") -> None:
    torch.save(state, filename)

def save_model(model: nn.Module, save_path: str) -> None:
    torch.save(model.state_dict(), save_path)

def load_model(model: nn.Module, load_path: str, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_output(data: np.ndarray, save_path: str, file_name: str = 'reconstructed.npy') -> None:
    os.makedirs(save_path, exist_ok=True)
    full_save_path = os.path.join(save_path, file_name)
    np.save(full_save_path, data)
    print(f'Model output is saved as numpy array to {full_save_path}')

