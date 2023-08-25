import chess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
letter_to_num = {v: k for k, v in num_to_letter.items()}

pieces = ['p', 'r', 'n', 'b', 'q', 'k']


def board_to_rep(board: chess.Board):
    layers = np.zeros((len(pieces), 8, 8))
    board_dic = board.piece_map()
    for pos, piece in board_dic.items():
        row = 7 - pos // 8
        column = pos % 8
        i = pieces.index(piece.symbol().lower())
        layers[i, row, column] = 1 if piece.symbol().isupper() else -1  # 1 for white, -1 for black
    return layers


class Chess_Dataset_white(Dataset):
    def __init__(self, positions, targets):
        self.targets = targets
        self.positions = positions

    def __len__(self):
        return np.shape(self.targets)[0]

    def __getitem__(self, idx):
        index = idx
        while abs(self.targets[index, 0]) > 999000: # exclude the mates
            index -= 1
        if self.targets[index, 1]:
            target = self.targets[index, 0]
            position = self.positions[index, :, :, :]

        elif index > 0:
            target = self.targets[index - 1, 0]
            position = self.positions[index - 1, :, :, :]
        else:
            target = self.targets[index + 1, 0]
            position = self.positions[index + 1, :, :, :]
        sample = {"position": position, "target": target}
        return sample


def collate_batch(batch):
    return torch.tensor([item['target'] for item in batch]), torch.stack(
        [torch.tensor(item['position']) for item in batch])


def get_samples_white(input, labels, batch_size):
    dataset = Chess_Dataset_white(input, labels)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
