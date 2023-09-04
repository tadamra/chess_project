#IMPORTS
import io
import tkinter as tk

import cairosvg
import chess
import chess.svg
import torch
from PIL import Image, ImageTk

from util import load_model, choose_move

# SPECIFY DEFAULT SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

class ChessGUI:
    def __init__(self, root, ai_move_function, board):
        self.root = root
        self.root.title("Chess Game")

        self.board = board
        self.ai_move_function = ai_move_function

        self.model = load_model()

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.update_board()

        self.root.bind("<Button-1>", self.on_board_click)
        self.state = 0
        self.from_square = None
        self.to_square = None


    def update_board(self):
        board_svg = chess.svg.board(board=self.board)
        img_bytes = cairosvg.svg2png(bytestring=board_svg)
        img = Image.open(io.BytesIO(img_bytes))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)


    def on_board_click(self, event):
        if not self.board.is_game_over():
            if self.state == 0:
                col = event.x // 50
                row = 7 - event.y // 50
                self.from_square = chess.square(col, row)
                self.state = 1
            elif self.state == 1:
                col = event.x // 50
                row = 7 - event.y // 50
                self.to_square = chess.square(col, row)
                legal_moves = [move for move in self.board.legal_moves if move.from_square == self.from_square and move.to_square == self.to_square]
                if len(legal_moves) > 0:
                    move = legal_moves[0]  # You can implement a way for the player to select the move
                    self.board.push(move)
                    self.update_board()
                    self.state = 2
                    print("Player moved:", move)
                    print("state:", self.state)
                else:
                    self.state = 0
            elif self.state == 2:
                print("AI is thinking...")
                if not self.board.is_game_over() and self.board.turn == chess.BLACK:
                    ai_move = self.ai_move_function(self.model, self.board, chess.BLACK)
                    self.board.push(ai_move)
                    self.update_board()
                self.state = 0
            else:
                print("Invalid state")
            if self.board.is_game_over():
                print("Game Over")
                print("Result:", self.board.result())

if __name__ == "__main__":
    root = tk.Tk()
    chess_gui = ChessGUI(root, choose_move)
    root.mainloop()




    

