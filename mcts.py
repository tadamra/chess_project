import chess
import chess.engine
import math
import random
import numpy as np

#set numpy seed
np.random.seed(0)

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
    def __init__(self, board, iterations=10):
        self.iterations = iterations
        self.root_node = Node(board)
    
    def mcts(self):
        for _ in range(self.iterations):
            not_fully_expanded_node = self.select(self.root_node) # while all children are fully expanded, select the best child. If not all children are fully expanded, return the node
            new_node = self.expand(not_fully_expanded_node) # expand the selected node
            simulation_result = self.simulate(new_node) # simulate a random game from the selected node
            self.backpropagate(new_node, simulation_result) # update the score of all nodes in the path from the selected node to the root node

        best_move = self.get_best_move(self.root_node)
        return best_move
    
    def ucb_score(self, parent_visits, child_visits, child_score, exploration_weight=1.0):
        if child_visits == 0:
            return float("inf")
        return child_score / child_visits + exploration_weight * math.sqrt(math.log(parent_visits) / child_visits)

    #def select(self, node,): #TODO do not necessarily expand each possible move, but use information on good move to expand earlier
    #    while not node.state.is_game_over():
    #        if not node.is_fully_expanded():
    #            return self.expand(node)
    #        else:
    #            node = self.best_child(node)
    #    return node

    def select(self, node, ): #TODO do not necessarily stop as soon as a node with unexpanded children is found, but explore the tree further
        while not node.state.is_game_over():
            if not node.is_fully_expanded():
                return node
            else:
                node = self.best_child(node)
        return node

    #def expand(self, node): # TODO: do not expand all untried moves before selecting one, but select one and expand it
    #    legal_moves = list(node.state.legal_moves)
    #    untried_moves = [move for move in legal_moves if move not in [child.state for child in node.children]]
    #    if untried_moves:
    #        move = random.choice(untried_moves)
    #        new_state = node.state.copy()
    #        new_state.push(move)
    #        child = Node(new_state, parent=node)
    #        node.children.append(child)
    #        return child
    #    else:
    #        return self.best_child(node)
        
    #def expand(self, node, exploration_prob = 0.5): # TODO: do not expand all untried moves before selecting one, but select one and expand it
    #    legal_moves = list(node.state.legal_moves)
    #    untried_moves = [move for move in legal_moves if move not in [child.state for child in node.children]]
    #    if (len(node.children) == 0)  or (untried_moves and np.random.rand() < exploration_prob):
    #        move = random.choice(untried_moves)
    #        new_state = node.state.copy()
    #        new_state.push(move)
    #        child = Node(new_state, parent=node)
    #        node.children.append(child)
    #        return child
    #    else:
    #        tmp_node = self.best_child(node)
    #        while tmp_node.is_fully_expanded() and not tmp_node.state.is_game_over():
    #            tmp_node = self.best_child(tmp_node)
    #        return tmp_node
        
    def expand(self, node,): # TODO: do not expand all untried moves before selecting one, but select one and expand it
        legal_moves = list(node.state.legal_moves)
        untried_moves = [move for move in legal_moves if move not in [child.state for child in node.children]]
        if untried_moves:
            move = random.choice(untried_moves)
            new_state = node.state.copy()
            new_state.push(move)
            child = Node(new_state, parent=node)
            node.children.append(child)
            return child
        else:
            return node

    def best_child(self, node):
        exploration_weight = 1.0
        children_scores = [self.ucb_score(node.visits, child.visits, child.score, exploration_weight) for child in node.children]
        return node.children[children_scores.index(max(children_scores))]

    def simulate(self, node, num_of_simulation): # TODO  do not simulate a random game, but use a neural network to predict the winner
        # For simplicity, we'll perform a random simulation
        sim_state = node.state.copy()
        while not sim_state.is_game_over():
            random_move = random.choice(list(sim_state.legal_moves))
            sim_state.push(random_move)
        return self.get_winner(sim_state)

    def get_winner(self, state):
        if state.is_checkmate():
            if state.turn == chess.WHITE: # white looses
                return -1
            else:
                return 1 # black looses
        return 0 # draw

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if node.state.turn == chess.WHITE:
                if result == 1:
                    node.score += 1
                elif result == -1:
                    pass
                elif node.score == 0:
                    node.score += 0.5
            elif node.state.turn == chess.BLACK:
                if result == 1:
                    pass
                elif result == -1:
                    node.score += 1
                elif node.score == 0:
                    node.score += 0.5
            node = node.parent

    def get_best_move(self, node):
        children_scores = [(child.score / child.visits, child.state) for child in node.children]
        _, best_state = max(children_scores, key=lambda x: x[0])
        legal_moves = list(node.state.legal_moves)
        assert best_state.peek() in legal_moves
        best_move = best_state.peek()
        return best_move
    
    def update_root(self, move):
        self.root_node = [child for child in self.root_node.children if child.state.peek() == move][0]

if __name__ == "__main__":
    board = chess.Board()
    mcts = MCTS(board, iterations=50)
    while True:
        print(mcts.root_node.state)
        print("##########")
        if mcts.root_node.state.is_game_over():
            break
        best_move = mcts.mcts()
        mcts.update_root(best_move)
        print(f"The best move found by MCTS is: {best_move}") 
    
