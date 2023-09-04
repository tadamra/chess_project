import pathlib
import tkinter as tk

import numpy as np
import torch

import chess_playing_GUI
from positionEvaluation import EvaluationCNN
from util import choose_move, sample_to_board

hidden_size = 128
hidden_layers = 4
i = 1
model_for_white = EvaluationCNN.EvaluationNetwork(hidden_layers=hidden_layers, hidden_size=hidden_size)
model_path = pathlib.Path(__file__).parent.resolve().parent / "positionEvaluation" / "model_white_weights.pth"
print(model_path)
model_for_white.load_state_dict(torch.load(model_path))

if __name__ == '__main__':
    start = 298
    targets = np.load(f"{pathlib.Path().resolve().parent}/data/labels{0}.npy",
                      allow_pickle=True)
    data = np.load(f"{pathlib.Path().resolve().parent}/data/positions{0}.npy",
                   allow_pickle=True)
    outputs = model_for_white(torch.tensor(data[start:start + 100]))
    for i in range(100):
        if targets[start + i][1]:
            root = tk.Tk()
            chess_playing_GUI.ChessGUI(root, choose_move, sample_to_board(data[start + i]))
            print(targets[start + i])
            print(outputs[i])
            root.mainloop()
