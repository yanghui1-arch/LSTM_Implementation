import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    
    def __init__(self, hidden_size, output_size) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        return self.linear(x)