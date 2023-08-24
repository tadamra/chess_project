import numpy as np
import pandas as pd
import chess
import gc
import re
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from processGame import board_to_rep

num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5:'f', 6:'g', 7:'h'}
letter_to_num = {v: k for k, v in num_to_letter.items()}

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
        self.linear1 = nn.Linear(hidden_size, 128)
        self.module_list = nn.ModuleList([CopiedSubModule(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Linear(128,1)
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for i in range(self.hidden_layers):
            x = self.module_list[i](x)
        x = x.view(x.size(0),)
        x = self.linear1()
        x = F.relu(x)
        x = self.output_layer(x)
        return x
