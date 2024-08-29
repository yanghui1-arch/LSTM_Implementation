from model.pure_lstm import PureLSTM
import torch
import torch.nn as nn


input_size = 64
hidden_size = 128
output_size = 4
x = torch.Tensor(2, input_size)
model = PureLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_cells=5)

print(f'model(x) = {model(x)}')