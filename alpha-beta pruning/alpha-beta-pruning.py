import random
import math

class TicTacToe(object):
    # Combinations for which the game is over
    winning_combos = (
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6])

    # Grid initialization
    def __init__(self, cells=[]):
        self.cells = [None for i in range(9)]

    # Show the grid
    def show(self):
        for element in [self.cells[i:i + 3] for i in range(0, len(self.cells), 3)]:
            print(element)
        print('\n')

    # Return the moves which are still available, i.e. ones that are equal to None
    def available_moves(self):
        return [index for index, cell_content in enumerate(self.cells) if cell_content is None]

    # Check if the game is over
    def finished(self):
        if None not in [cell_content for cell_content in self.cells]:
            return True
        if self.winner() != None:
            return True
        return False

    # Use previous defined 'winning_combos' to return the winner
    def winner(self):
        for player in ('X', 'O'):
            positions = self.get_cells(player)
            for combo in self.winning_combos:
                win = True
                for pos in combo:
                    if pos not in positions:
                        win = False
                if win:
                    return player
        return None

    # Return the indexes where the player has made a move
    def get_cells(self, player):
        return [index for index, cell_content in enumerate(self.cells) if cell_content == player]

    # Perform a move, i.e. setting the cell's value to X or O according to the player variable
    def make_move(self, position, player):
        self.cells[position] = player

    # Alpha-Beta pruning method
    def alphabeta(self, node, player, alpha, beta):
        if node.finished():
            if self.winner() == 'X' : return -10
            elif self.winner() is None : return 0
            elif self.winner() == 'O' : return 10

        if player == 'O':
            val = alpha
            for move in node.available_moves():
                node.make_move(move, player)
                val = max(val, self.alphabeta(node, get_enemy(player), alpha, beta))
                node.make_move(move, None)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break # beta cut-off --> MAX would never choose a value < previous found value, so there is no point in looking at the other successor states
            return val
        else:
            val = beta
            for move in node.available_moves():
                node.make_move(move, player)
                val = min(val, self.alphabeta(node, get_enemy(player), alpha, beta))
                node.make_move(move, None)
                beta = min(beta, val)
                if alpha >= beta:
                    break # alpha cut-off --> see above
            return val

def best_move(board, player):
    a = -2
    moves = []

    if len(board.available_moves()) == 9:
        return random.choice([1, 3, 5, 7])

    for move in board.available_moves():
        board.make_move(move, player)
        val = board.alphabeta(board, get_enemy(player), -2, 2)
        board.make_move(move, None)
        if val > a: # If val is the best value found till now, 'moves' will contain only this move
            a = val
            moves = [move]
        elif val == a: # If the two values are equal, the move is not the best move we can do, there are other moves which bring to the same result
            moves.append(move)
    return random.choice(moves)

def get_enemy(player):
    return 'O' if player == 'X' else 'X'

if __name__ == "__main__":
    board = TicTacToe()
    board.show()

    while not board.finished():
        player = 'X'
        
        player_move = int(input("Next Move [1 - 9]: ")) - 1
        if not player_move in board.available_moves():
            continue
        
        board.make_move(player_move, player)
        board.show()
        if board.finished():
            break

        player = get_enemy(player)
        ai_move = best_move(board, player)
        board.make_move(ai_move, player)
        board.show()

    print("The winner is", board.winner())