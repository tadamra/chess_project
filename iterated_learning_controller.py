from mcts_deep_learning import MCTS
import chess
import chess.engine
import math
import random
import numpy as np
import pandas as pd
from util import get_letter_to_num, board_to_rep
from neural_networks import load_policy_model, load_eval_model
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    board = chess.Board()
    policy_model = load_policy_model()
    eval_model = load_eval_model()
    res = eval_model.forward(torch.ones(1,6,8,8)*-1)
    out_1 =policy_model.forward(torch.ones(1,6,8,8)*-1)
    out_2 = policy_model.forward(torch.ones(1,6,8,8))
    print(out_1)
    print(out_2)
    print(res)
    exit()
    res = eval_model(torch.ones(1,6,8,8))
    print(res)
    exit()
    mcts = MCTS(board, policy_model, eval_model, iterations= 50)
    root_node = mcts.root_node
    i = 0
    winner = 0
    states = []
    moves = []
    while True:
        if i>= 1000:
            print("i>=1000")
            break
        print(mcts.root_node.state)
        print("##########")
        if mcts.root_node.state.is_game_over():
            # get reason for game over
            outcome = mcts.root_node.state.outcome()
            print("outcome", outcome)
            winner = mcts.get_winner(mcts.root_node.state)
            print("winner", winner)
            break
        mcts.mcts()
        best_move = mcts.get_best_move(mcts.root_node)
        if len(moves) >= 6:
            if str(moves[-4]) == str(best_move): #and moves[-2] == moves[-6]:
                best_move = mcts.get_best_move(mcts.root_node, i = 1)
                print("repetition avoidance")
        states.append(mcts.root_node.state.board_fen())
        moves.append(str(best_move))
        mcts.update_root(best_move)
        print(f"The best move found by MCTS is: {best_move}")
        i+=1
    training_data_policy = zip(states, moves)
    training_data_eval = zip(states, np.full(len(states), winner))
    policy_df = pd.DataFrame(training_data_policy, columns=["FEN", "move"])
    eval_df = pd.DataFrame(training_data_eval, columns=["FEN", "label"])
    policy_df.to_csv("policy_training_data.csv")
    eval_df.to_csv("eval_training_data.csv")
    #network.train(training_data_policy, training_data_eval)

    #mcts.mcts()
    #eval_training_data = mcts.collect_eval_training_data()
    #policy_training_data = mcts.collect_policy_training_data()
    #for i in range(len(eval_training_data)):
    #    print("#####################")
    #    print("evaluation", eval_training_data[i][1])
    #    print(eval_training_data[i][0])
    #labelled = MCTS.preprocess_training_data(policy_training_data)