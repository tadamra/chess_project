#IMPORTS
import numpy as np
import chess
import torch
import torch.nn.functional as F
import chess.svg
import tkinter as tk
from PIL import Image, ImageTk
import cairosvg
import io
from util import load_model, choose_move

# SPECIFY DEFAULT SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

class ChessGame:
    def __init__(self, ai_move_function):

        self.board = chess.Board()
        self.ai_move_function = ai_move_function

        self.model = load_model()

        self.state = 0
        self.from_square = None
        self.to_square = None

    def run(self):
        while not self.board.is_game_over():
            if self.state == 0:
                print("AI1 is thinking...")
                if not self.board.is_game_over() and self.board.turn == chess.WHITE:
                    ai_move = self.ai_move_function(self.model, self.board, chess.WHITE)
                    self.board.push(ai_move)
                self.state = 2
            elif self.state == 2:
                print("AI2 is thinking...")
                if not self.board.is_game_over() and self.board.turn == chess.BLACK:
                    ai_move = self.ai_move_function(self.model, self.board, chess.BLACK)
                    self.board.push(ai_move)
                self.state = 0
            else:
                print("Invalid state")
            if self.board.is_game_over():
                print("Game Over")
                print("Result:", self.board.result())

if __name__ == "__main__":
    chess_game = ChessGame(choose_move)
    chess_game.run()




    

