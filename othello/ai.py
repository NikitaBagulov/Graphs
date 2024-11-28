import math
import numpy as np
from graphviz import Digraph

DEPTH = 4

class OthelloAI:
    def __init__(self, game_state, current_player):
        self.game_state = np.array(game_state) 
        self.current_player = current_player
        self.opponent = 'black' if current_player == 'white' else 'white'
        self.depth = DEPTH
        self.dot = Digraph()


    def is_valid_move(self, x, y):
        if self.game_state[x][y] is not None:
            return False

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False

            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.game_state[nx][ny] == self.opponent:
                    found_opponent = True
                elif self.game_state[nx][ny] == self.current_player and found_opponent:
                    return True
                else:
                    break
                nx += dx
                ny += dy

        return False

    def make_move(self, game_state, x, y, player):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        game_state[x][y] = player

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            tiles_to_flip = []

            while 0 <= nx < 8 and 0 <= ny < 8:
                if game_state[nx][ny] == self.opponent:
                    tiles_to_flip.append((nx, ny))
                elif game_state[nx][ny] == player:
                    for px, py in tiles_to_flip:
                        game_state[px][py] = player
                    break
                else:
                    break

                nx += dx
                ny += dy

    def is_game_over(self, game_state):
        for player in [self.current_player, self.opponent]:
            for x in range(8):
                for y in range(8):
                    if self.is_valid_move_for_player(game_state, x, y, player):
                        return False
        return True

    def is_valid_move_for_player(self, game_state, x, y, player):
        if game_state[x][y] is not None:
            return False

        opponent = 'black' if player == 'white' else 'white'
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False

            while 0 <= nx < 8 and 0 <= ny < 8:
                if game_state[nx][ny] == opponent:
                    found_opponent = True
                elif game_state[nx][ny] == player and found_opponent:
                    return True
                else:
                    break
                nx += dx
                ny += dy

        return False


    def visualize_board(self, game_state, score, node_id, move_label=None):
        transposed_board = np.transpose(game_state)

        board_str = '\n'.join([' '.join(['_' if cell is None else cell[0] for cell in row]) for row in transposed_board])

        # Highlight the move if specified
        if move_label:
            label = f"Score: {score}\n{board_str}\n**{move_label}**"
            color = "lightblue"  # Color for the current player's move
        else:
            label = f"Score: {score}\n{board_str}"
            color = "white"  # Default color for other nodes
        
        self.dot.node(node_id, label=label, style='filled', fillcolor=color)

    def get_best_move(self):
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        valid_moves = [(x,y) for x in range(8) for y in range(8) if self.is_valid_move(x,y)]

        root_node_id = "root"
        self.visualize_board(self.game_state, 0, root_node_id)

        for x,y in valid_moves:
            temp_game_state = np.copy(self.game_state)
            self.make_move(temp_game_state,x,y,self.current_player)
            score = self.minimax(temp_game_state,self.depth - 1, alpha, beta, True, root_node_id + f"_{x}_{y}")
            print(f"Move {x+1},{y+1} -> Score: {score}") if best_move is not None else print("None moves")
            
            if score > best_score:
                best_score = score
                best_move = (x,y)
            
            alpha = max(alpha ,score)
            if beta <= alpha:
                break
        self.dot.render('othello_tree', format='svg', cleanup=True)
        return best_move

    def minimax(self, game_state, depth, alpha, beta, is_maximizing, parent_node_id):
        if depth == 0 or self.is_game_over(game_state):
            score = self.evaluate_board(game_state)
            return score

        if is_maximizing:
            max_eval = -math.inf
            valid_moves = [(x, y) for x in range(8) for y in range(8) if self.is_valid_move(x, y)]
            for x, y in valid_moves:
                temp_game_state = np.copy(game_state)
                self.make_move(temp_game_state, x, y, self.current_player)
                node_id = f"{parent_node_id}_max_{x}_{y}"
                eval = self.minimax(temp_game_state, depth - 1, alpha, beta, False, node_id)
                self.visualize_board(temp_game_state, eval, node_id, move_label=f"MOVED: {x+1},{y+1}")
                self.dot.edge(parent_node_id, node_id)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    pruned_node_id = f"{node_id}_pruned"
                    self.dot.node(pruned_node_id, label=f"Pruned\nAlpha: {alpha}, Beta: {beta}", style='filled', fillcolor='lightcoral')
                    self.dot.edge(parent_node_id, pruned_node_id)
                    break
            return max_eval
        else:
            min_eval = math.inf
            valid_moves = [(x, y) for x in range(8) for y in range(8) if self.is_valid_move(x, y)]
            for x, y in valid_moves:
                temp_game_state = np.copy(game_state)
                self.make_move(temp_game_state, x, y, self.opponent)
                node_id = f"{parent_node_id}_min_{x}_{y}"
                eval = self.minimax(temp_game_state, depth - 1, alpha, beta, True, node_id)
                
                self.visualize_board(temp_game_state, eval, node_id, move_label=f"MOVED: {x+1},{y+1}")
                self.dot.edge(parent_node_id, node_id)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    pruned_node_id = f"{node_id}_pruned"
                    self.dot.node(pruned_node_id, label=f"Pruned\nAlpha: {alpha}, Beta: {beta}", style='filled', fillcolor='lightcoral')
                    self.dot.edge(parent_node_id, pruned_node_id)
                    break
            return min_eval

    def evaluate_board(self ,game_state):
        score = np.sum(game_state == self.current_player) - np.sum(game_state == self.opponent)
        return score
