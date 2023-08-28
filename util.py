import numpy as np
import chess
import torch
import torch.nn.functional as F

# SPECIFY DEFAULT SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5:'f', 6:'g', 7:'h'}
letter_to_num = {v: k for k, v in num_to_letter.items()}

def get_num_to_letter(num):
    return num_to_letter[num]

def get_letter_to_num(letter):
    return letter_to_num[letter]

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

def move_to_rep(move, board):
    board.push_san(move).uci( )
    move = str(board.pop())
    from_output_layer = np. zeros( (8,8) )
    from_row = 8-int(move[1])
    from_column = letter_to_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np. zeros( (8,8) )
    to_row = 8-int(move[3])
    to_column = letter_to_num[move[2]]
    to_output_layer[to_row, to_column] = 1
    return np.stack( (from_output_layer, to_output_layer), axis = 0 )

#def choose_move(model, board, color):
#    legal_moves = list(board.legal_moves)
#    x = torch.Tensor(board_to_rep(board)).double().to(device)
#    if color == chess.BLACK:
#        x *= -1
#    x = x.unsqueeze(0)
#    move = model(x)
#    move = move.squeeze()
#    vals = []
#    froms = [str(legal_move)[0:2] for legal_move in legal_moves]
#    froms = list(set(froms))
#    for from_ in froms:
#        val = move[0, ...][8-int(from_[1]), get_letter_to_num[from_[0]],]
#        vals.append(val)
#    probs = F.softmax(torch.Tensor(vals), dim = 0)
#    chosen_from = str(np.random.choice(froms, size=1, p = probs.detach().numpy())[0])[:2]
#    vals = []
#    for legal_move in legal_moves:
#        from_ = str(legal_move)[0:2]
#        if from_ == chosen_from:
#            to = str(legal_move)[2:4]
#            val = move[1, ...][8-int(to[1]), get_letter_to_num[to[0]],]
#            vals.append(val.item())
#        else:
#            vals.append(0)
#    chosen_move = legal_moves[np.argmax(vals)]
#    return chosen_move

def get_model_output(model, board, color):
    x = torch.Tensor(board_to_rep(board)).float().to(device)
    if color == chess.BLACK:
        x *= -1
        x = x.flip(1)
    x = x.unsqueeze(0)
    move = model(x.double())
    move = move.squeeze()
    move = move.flip(1)
    return move

def choose_move(model, board, color):
    strategy = "random" # "random" or "best"

    move = get_model_output(model, board, color)
    legal_moves = list(board.legal_moves)
    vals = []
    froms = [str(legal_move)[0:2] for legal_move in legal_moves]
    froms = list(set(froms))

    # get predicted values for valid from squares
    for from_ in froms:
        val = move[0, ...][8-int(from_[1]), letter_to_num[from_[0]],]
        vals.append(val)
    
    # normalize values to probabilities
    probs = F.softmax(torch.Tensor(vals), dim = 0)

    # choose from square based on probabilities
    if strategy == "best":
        chosen_from = froms[np.argmax(probs.detach().numpy())][:2]
    elif strategy == "random":
        chosen_from = str(np.random.choice(froms, size=1, p = probs.detach().numpy())[0])[:2]
    else:
        raise ValueError("strategy must be 'best' or 'random'")

    # get predicted values for valid to squares for chosen from square
    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[0:2]
        if from_ == chosen_from:
            to = str(legal_move)[2:4]
            val = move[1, ...][8-int(to[1]), letter_to_num[to[0]],]
            vals.append(val.item())
        else:
            vals.append(0)
    # normalize values to probabilities
    probs = F.softmax(torch.Tensor(vals), dim = 0)

    if strategy == "best":
        chosen_move = legal_moves[np.argmax(probs.detach().numpy())]
    elif strategy == "random":
        chosen_move = np.random.choice(legal_moves, size=1, p = probs.detach().numpy())[0]
    else:
        raise ValueError("strategy must be 'best' or 'random'")
    return chosen_move

#def generate_sequence_by_AI(model,):
#  model.to(device)
#  board = chess.Board()
#  i=0
#  moves = []
#  while (not board.is_game_over()) and i<200 :
#    move = choose_move(model, board, board.turn)
#    board.push(move)
#    moves.append(move)
#    i+=1
#  return moves
#
#def display_sequence_by_AI(model):
#    moves = generate_sequence_by_AI(model)
#    board = chess.Board()
#    print(board)
#    for move in moves:
#      board.push(move)
#      print(board)