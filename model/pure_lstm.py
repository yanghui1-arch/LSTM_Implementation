import torch
import torch.nn as nn
from layers.input_layer import InputLayer
from layers.LSTM_layer import LSTMLayer
from layers.output_layer import OutputLayer

class PureLSTM(nn.Module):

    def __init__(self, input_size, num_cells, hidden_size, output_size) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_layer = LSTMLayer(self.input_layer.out_features, hidden_size=hidden_size, num_cells=num_cells)
        self.output_layer = OutputLayer(hidden_size, output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        hidden_state = self.lstm_layer(x)
        return self.output_layer(hidden_state)