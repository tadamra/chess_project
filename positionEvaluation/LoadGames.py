import pathlib

import chess.pgn
import numpy as np

from processGame import board_to_rep


def generate_data(filename, number_of_games_to_analyze=np.inf):
    elo_labels = []
    labels = []
    positions = []
    packet_number = 0
    pgn = open(f"{pathlib.Path().resolve()}/{filename}")
    counter = 0
    while (game := chess.pgn.read_game(pgn)) is not None and counter < number_of_games_to_analyze:
        counter +=1
        board = game.board()
        node = game
        for move in game.mainline_moves():
            node = node.next()
            board.push(move)
            if node.eval() is None:
                break
            else:
                labels.append([node.eval().relative.score(mate_score=1000000), node.eval().turn])
                positions.append(board_to_rep(board))
                if len(positions) == 20000:
                    np.save(f"data/labels{packet_number}", np.array(labels))
                    np.save(f"data/positions{packet_number}", np.array(positions))
                    packet_number += 1
                    labels = []
                    positions = []
    np.save(f"data/labels{packet_number}", np.array(labels))
    np.save(f"data/positions{packet_number}", np.array(positions))
    # a = np.load(f"{pathlib.Path().resolve()}/labels.npy", allow_pickle=True)
    # b = np.load(f"{pathlib.Path().resolve()}/positions.npy", allow_pickle=True)
    # print(a)
    # print(b)


if __name__ == '__main__':
    # DB used here is lichess_db_standard_rated_2016-03.pgn that can be easily downloaded on lichess database
    # whose website is : https://database.lichess.org/
    generate_data("lichess_db_standard_rated_2016-03.pgn")
