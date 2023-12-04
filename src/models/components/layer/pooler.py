import torch
import torch.nn as nn


class AvgPooler(nn.Module):
    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.mean(x, dim=self.dim)
