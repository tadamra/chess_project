import numpy as np
import chess
import torch
import torch.nn.functional as F
import chess.svg
import tkinter as tk
from neural_networks import CopiedChessNetwork

# SPECIFY DEFAULT SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5:'f', 6:'g', 7:'h'}
letter_to_num = {v: k for k, v in num_to_letter.items()}

def load_model():
    hidden_size = 128
    hidden_layers = 4
    model = CopiedChessNetwork(hidden_layers, hidden_size).to(device)
    model.load_state_dict(torch.load('chess_project/model_weights.pth'))
    return model

def board_to_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = np.zeros((len(pieces), 8, 8))
    board_dic = board.piece_map()
    for pos, piece in board_dic.items():
        row = 7 - pos // 8
        column = pos % 8
        i = pieces.index(piece.symbol().lower())
        layers[i, row, column] = 1 if piece.symbol().isupper() else -1 #1 for white, -1 for black
    return layers

def choose_move(model, board, color):
    legal_moves = list(board.legal_moves)
    x = torch.Tensor(board_to_rep(board)).float().to(device)
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)
    move = model(x.double())
    move = move.squeeze()
    vals = []
    froms = [str(legal_move)[0:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for from_ in froms:
        val = move[0, ...][8-int(from_[1]), letter_to_num[from_[0]],]
        vals.append(val)
    probs = F.softmax(torch.Tensor(vals), dim = 0)
    chosen_from = str(np.random.choice(froms, size=1, p = probs.detach().numpy())[0])[:2]
    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[0:2]
        if from_ == chosen_from:
            to = str(legal_move)[2:4]
            val = move[1, ...][8-int(to[1]), letter_to_num[to[0]],]
            vals.append(val.item())
        else:
            vals.append(0)
    chosen_move = legal_moves[np.argmax(vals)]
    return chosen_move