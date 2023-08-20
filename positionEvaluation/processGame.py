import chess
import numpy as np

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
