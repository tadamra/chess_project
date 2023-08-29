import chess
import chess.engine
import math
import random
import numpy as np
import pandas as pd
from util import get_letter_to_num, board_to_rep, get_probs_from_extended_rep, get_probs_from_rep
from neural_networks import preprocess_board_for_policy, postprocess_actions_for_policy, preprocess_board_for_eval, postprocess_value_for_eval
import torch
import torch.nn.functional as F

#set numpy seed
np.random.seed(42)
# SPECIFY DEFAULT SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))

class MCTS():
    # class invariant: root_node.state must not be a terminal state
    def __init__(self, board, policy_model, eval_model, iterations=10):
        self.iterations = iterations
        self.root_node = Node(board)
        self.root_node.visits = 1
        self.eval_model = eval_model
        self.policy_model = policy_model
    
    def mcts(self):
        for _ in range(self.iterations):
            #if _ % 5 == 0:
            #    print("MCTS iteration", _)
            leaf_node, action = self.select(self.root_node)
            new_node = self.expand(leaf_node, action)
            simulation_result = self.simulate(new_node)
            self.backpropagate(new_node, simulation_result)
    
    def select_action(self, node,):
        inp = preprocess_board_for_policy(node.state)
        output = self.policy_model(inp)
        actions = postprocess_actions_for_policy(output, node.state)

        legal_moves = list(node.state.legal_moves)
        #probs =np.zeros(len(legal_moves))
        #probs_from = np.zeros(len(legal_moves))
        #probs_to = np.zeros(len(legal_moves))
        #for i, legal_move in enumerate(legal_moves):
        #    from_ = str(legal_move)[0:2]
        #    to = str(legal_move)[2:4]
        #    probs_from[i] = actions[0, ...][8-int(from_[1]), get_letter_to_num(from_[0]),] # Attention: probs_from might contain probabilities for the same from_ multiple times
        #    probs_to[i] = actions[1, ...][8-int(to[1]), get_letter_to_num(to[0]),] # Attention: probs_to might contain probabilities for the same to multiple times
        #probs_from = F.softmax(torch.Tensor(probs_from), dim = 0)
        #probs_to = F.softmax(torch.Tensor(probs_to), dim = 0)
        #probs = probs_from * probs_to # Attention: probs are not actually probabilities
        #probs = probs / probs.sum()
        probs = get_probs_from_rep(actions, node.state)
        #print(node.state)
        #print(legal_moves)
        #print(probs)
        #input("Press Enter to continue...")
        index = np.argmax(self.ucb_q_scores(node, probs))
        return legal_moves[index]

    def ucb_q_scores(self, node, probs,exploration_weight=0.5):
        legal_moves = list(node.state.legal_moves)
        q_scores = np.zeros(len(legal_moves))
        for i, legal_move in enumerate(legal_moves):
            next_state = node.state.copy()
            next_state.push(legal_move)
            #value, _ = self.nn(next_state) # Note: currently, value is always zero because DummyNetwork is used #TODO use value network only instead of simulation but not here
            next_state_visits = self.get_child_visits(node, next_state)
            next_state_score = self.get_child_score(node, next_state)
            value = next_state_score/next_state_visits if next_state_visits != 0 else 0
            if node.visits == 0:
                print("node.visits == 0")
            q_scores[i] =  value  + exploration_weight * probs[i] * (np.sqrt(node.visits-1)/(1+next_state_visits))
        return q_scores


    def get_child_visits(self, node, next_state):
        for child in node.children:
            if child.state == next_state:
                return child.visits
        return 0
    
    def get_child_score(self, node, next_state):
        for child in node.children:
            if child.state == next_state:
                return child.score
        return 0

    def select(self, node,):
        while not node.state.is_game_over():
            action = self.select_action(node)
            already_tested_action = [child for child in node.children if child.state.peek() == action]
            if len(already_tested_action) != 0:
                node = already_tested_action[0]
            else:
                return node, action
        return node, None
        
    def expand(self, node, action): 
        if action is None:
            return node
        new_state = node.state.copy()
        new_state.push(action)
        child = Node(new_state, parent=node)
        node.children.append(child)
        return child

    def simulate(self, node,): 
        sim_state = node.state.copy()
        i = 0
        while not sim_state.is_game_over():
            if i > 100:
                break
            random_move = random.choice(list(sim_state.legal_moves))
            sim_state.push(random_move)
        winner = self.get_winner(sim_state)
        return winner
    
    # use when we have a neural network to evaluate a position
    #def simulate(self, node,):
    #    sim_state = node.state.copy()
    #    if sim_state.is_game_over():
    #        return self.get_winner(sim_state)
    #    else:
    #        inp = preprocess_board_for_eval(sim_state)
    #        out = self.eval_model(inp)
    #        value = postprocess_value_for_eval(out, sim_state)
    #        print(node.state.turn, value)
    #        value = F.tanh(value)
    #        return value # positive value indicates white is winning, negative value indicates black is winning
    
    def get_winner(self, state):
        if state.is_checkmate():
            if state.turn == chess.WHITE:
                return -1 #black wins
            else:
                return 1 #white wins
        return 0 #draw

    def backpropagate(self, node, result):
        if node.state.turn == chess.BLACK:
            result *= -1
        while node is not None:
            node.visits += 1

            node.score += result
            result *= -1

            node = node.parent

    def get_best_move(self, node, iterations = 0):
        children_scores = [child.visits for child in node.children]

        # filter i best moves (to avoid repetition of moves)
        for _ in range(iterations):
            index = np.argmax(children_scores)
            children_scores[index] = -np.inf

        best_index = np.argmax(children_scores)
        best_state = node.children[best_index].state
        legal_moves = list(node.state.legal_moves)
        assert best_state.peek() in legal_moves
        best_move = best_state.peek()
        return best_move
    
    def update_root(self, move):
        self.root_node = [child for child in self.root_node.children if child.state.peek() == move][0]
    
    def create_label(policy_list):
        np_policy = np.zeros((2,8,8))
        for move, prob in policy_list:
            from_ = str(move)[0:2]
            to = str(move)[2:4]
            np_policy[0, 8-int(from_[1]), get_letter_to_num(from_[0])] = max(np_policy[0, 8-int(from_[1]), get_letter_to_num(from_[0])], prob)
            np_policy[1, 8-int(to[1]), get_letter_to_num(to[0])] = max(np_policy[1, 8-int(to[1]), get_letter_to_num(to[0])], prob)
        return np_policy
    
    def preprocess_training_data(policy_list):
        training_data = []
        for board, policy in policy_list:
            x = board_to_rep(board)
            y = MCTS.create_label(policy)
            x = torch.Tensor(x).double()
            y = torch.Tensor(y).double()
            if board.turn == chess.BLACK:
                x *= -1
                x = torch.flip(x, [1])
                y = torch.flip(y, [1])
            training_data.append((x,y))
        return training_data



    
