import torch
import torch.nn as nn

class InputLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        return self.linear(x)