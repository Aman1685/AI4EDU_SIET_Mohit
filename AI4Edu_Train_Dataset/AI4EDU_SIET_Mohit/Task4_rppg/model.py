# model.py

import torch
import torch.nn as nn

class MLP_PyTorch(nn.Module):
    """
    Simple fully connected MLP for classification.
    """
    def __init__(self, input_dim, num_classes, hidden1=128, hidden2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(pth_path, input_dim, num_classes, device='cpu'):
    """
    Load a trained MLP model from a .pth file.
    """
    model = MLP_PyTorch(input_dim, num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model
