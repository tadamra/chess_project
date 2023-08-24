from EvaluationCNN import CopiedChessNetwork
import torch.nn as nn
import torch.nn.functional as F
import torch

lr = 3e-4
hidden_size = 128
hidden_layers = 4
model = CopiedChessNetwork(hidden_layers=hidden_layers, hidden_size=hidden_size)
metric_from = nn.MSELoss()
metric_to = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)