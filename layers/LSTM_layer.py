import torch
import torch.nn as nn
from memory_cell.memory_cell import MemoryCell

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_cells):
        super().__init__()
        self.memory_cell = MemoryCell(input_size=input_size, hidden_size=hidden_size)
        self.lstm = nn.ModuleList([self.memory_cell for _ in range(num_cells)])

    def forward(self, x):
        for layer in self.lstm:
            x = layer(x)
        return x