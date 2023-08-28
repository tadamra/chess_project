import torch.nn as nn
import torch.nn.functional as F

num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
letter_to_num = {v: k for k, v in num_to_letter.items()}
board_size = 64

class CopiedSubModule(nn.Module):

    def __init__(self, hidden_size):
        super(CopiedSubModule, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
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


class EvaluationNetwork(nn.Module):

    def __init__(self, hidden_layers, hidden_size):
        super(EvaluationNetwork, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.input_layer.weight)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.linear1 = nn.Linear(board_size*hidden_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.dropout2 = nn.Dropout(0.2)
        self.module_list = nn.ModuleList([CopiedSubModule(hidden_size) for _ in range(hidden_layers)])
        self.dropout3 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)
        x = self.bn1(x)
        # Flatten the tensor for the linear layers
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, hidden_size * output_height * output_width)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.output_layer(x)

        return x


class ChessEvaluationNetwork(nn.Module):
    def __init__(self):
        super(ChessEvaluationNetwork, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


