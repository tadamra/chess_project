import numpy as np
from torch.utils.data import Dataset, DataLoader
import chess
from util import board_to_rep, move_to_rep, move_to_extended_rep
import re
import torch

class Copied_Dataset(Dataset):
  def __init__(self, games, length):
    super(Copied_Dataset, self).__init__()
    self.games = games
    self.length = length

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    game_i= np.random.randint(self.games.shape[0])
    random_game = self.games['AN'].values[game_i]
    moves = create_move_list(random_game)
    game_state_i = np.random.randint(len(moves)-1)
    next_move = moves [game_state_i]
    moves = moves[:game_state_i]
    board = chess.Board()
    for move in moves:
      board.push_san(move)
    x = board_to_rep(board)
    y = move_to_rep(next_move, board)
    x = torch.Tensor(x).double()
    y = torch.Tensor(y).double()
    if board.turn == chess.BLACK:
      x *= -1
      x = torch.flip(x, [1])
      y = torch.flip(y, [1])
    return x, y
  
class CopiedDatasetExtendedMoveRep(Dataset):
  def __init__(self, games, length):
    super(CopiedDatasetExtendedMoveRep, self).__init__()
    self.games = games
    self.length = length

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    game_i= np.random.randint(self.games.shape[0])
    random_game = self.games['AN'].values[game_i]
    moves = create_move_list(random_game)
    game_state_i = np.random.randint(len(moves)-1)
    next_move = moves [game_state_i]
    moves = moves[:game_state_i]
    board = chess.Board()
    for move in moves:
      board.push_san(move)
    x = board_to_rep(board)
    y = move_to_extended_rep(next_move, board)
    x = torch.Tensor(x).double()
    y = torch.Tensor(y).double()
    if board.turn == chess.BLACK:
      x *= -1
      x = torch.flip(x, [1])
      y = torch.flip(y, [1])
    return x, y
  
def create_move_list(AN_string):
    return re.sub(r'\d*\.', '', AN_string).split()

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


class RetrainPolicyDataset(Dataset):
    def __init__(self, df, ):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = chess.Board(row['FEN'])
        next_move = row['move']
        x = board_to_rep(board)
        y = move_to_rep(next_move, board)
        x = torch.Tensor(x).double()
        y = torch.Tensor(y).double()
        if board.turn == chess.BLACK:
            x *= -1
        x = torch.flip(x, [1])
        y = torch.flip(y, [1])
        return x, y

class RetrainEvalDataset(Dataset):
    def __init__(self, df, ):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = chess.Board(row['FEN'])
        winner = row['label']
        x = board_to_rep(board)
        y = winner
        if board.turn == chess.BLACK:
            x *= -1
            x = torch.flip(x, [1])
            y *= -1
        return x, y # 0 if draw, 1 if player at turn wins, -1 if player at turn loses

#class Chess_Dataset(Dataset):
#
#    def __init__(self, AN, length):
#        self.AN = AN
#        self.buffer = []
#        self.length = length
#
#    def __len__(self):
#        return self.length
#
#    def __getitem__(self, index):
#        while len(self.buffer) < 1000:
#            game_id = np.random.randint(self.AN.shape[0])
#            game = self.AN['AN'].iloc[game_id]
#            board = chess.Board()
#            move_list = create_move_list(game)
#            for i, move in enumerate(move_list[:-1]):
#                board_rep = board_to_rep(board)
#                move_rep = move_to_rep(move, board)
#                if i % 2 == 1:
#                    board_rep *= -1
#                self.buffer.append((board_rep, move_rep))
#                board.push_san(move)
#
#        return self.buffer[np.random.randint(0, len(self.buffer))]
    
#class Chess_Dataset(Dataset):
#    def __init__(self, AN,):
#        self.AN = AN
#        self.buffer = []
#        self.length = self.AN.shape[0] * 10 # where 10 is a safe lower bound estimate for average number of moves per game
#        self.index = 0
#
#    def __len__(self):
#        return self.length
#
#    def __getitem__(self, index):
#        if len(self.buffer) == 0:
#            game = self.AN['AN'].iloc[self.index]
#            self.index += 1
#            self.index = self.index % self.AN.shape[0]
#            board = chess.Board()
#            move_list = create_move_list(game)
#            for i, move in enumerate(move_list[:-1]):
#                board_rep = board_to_rep(board)
#                move_rep = move_to_rep(move, board)
#                if board.turn == chess.BLACK:
#                    board_rep *= -1
#                self.buffer.append((board_rep, move_rep))
#                board.push_san(move)
#        return self.buffer.pop()