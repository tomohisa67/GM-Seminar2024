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

