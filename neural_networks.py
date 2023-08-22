import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import board_to_rep
import chess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)


class CopiedSubModule(nn.Module):
    
    def __init__(self, hidden_size):
        super(CopiedSubModule, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding = 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_input
        x = self.activation2(x)
        return x
    
class CopiedChessNetwork(nn.Module):

    def __init__(self, hidden_layers, hidden_size):
        super(CopiedChessNetwork, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride = 1, padding = 1)
        self.module_list = nn.ModuleList([CopiedSubModule(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride = 1, padding = 1)
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)
        return x
    
class DummyNetwork(nn.Module):
    def __init__(self, policy_network):
        super(DummyNetwork, self).__init__()
        self.policy_network = policy_network

    def forward(self, board):
        x = board_to_rep(board)
        if board.turn == chess.BLACK:
            x *= -1
        x = torch.Tensor(x).double().to(device)
        x = x.unsqueeze(0)
        actions = self.policy_network(x)
        actions = actions.squeeze()
        value = 0
        return value, actions
    
def load_model():
    hidden_size = 128
    hidden_layers = 4
    model = CopiedChessNetwork(hidden_layers, hidden_size).to(device)
    model.load_state_dict(torch.load('chess_project/bad_model_weights.pth'))
    return model