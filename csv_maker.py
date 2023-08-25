import csv


def pretty_move(move, move_num):
    """
    This function convert the move name from the formate e.g. d4d6 to 1. d4 d6
    """
    square_1 = str(move)[0:2]
    square_2 = str(move)[2:3]
    return str(move_num) + ". " + square_1 + " " + square_2


def export_csv(moves: list):
    moves_str = ''.join(moves)
    with open('moves.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['AN'])
        writer.writerow([moves_str])
