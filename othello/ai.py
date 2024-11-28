import math
import numpy as np
from graphviz import Digraph

DEPTH = 4

class OthelloAI:
    def __init__(self, game_state, current_player, visualize=True):
        self.game_state = np.array(game_state) 
        self.current_player = current_player
        self.opponent = 'black' if current_player == 'white' else 'white'
        self.depth = DEPTH
        self.dot = Digraph()
        self.visualize_enabled = visualize
        self.corner_weights = [
            [100, -10, 10, 10, 10, 10, -10, 100],
            [-10, -20,  0,  0,  0,  0, -20, -10],
            [ 10,   0,  5,  5,  5,  5,   0,  10],
            [ 10,   0,  5,  5,  5,  5,   0,  10],
            [ 10,   0,  5,  5,  5,  5,   0,  10],
            [ 10,   0,  5,  5,  5,  5,   0,  10],
            [-10, -20,  0,  0,  0,  0, -20, -10],
            [100, -10, 10, 10, 10, 10, -10, 100]]

    def is_valid_move(self, x, y, player, opponent):
        if self.game_state[x][y] is not None:
            return False
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.game_state[nx][ny] == opponent:
                    found_opponent = True
                elif self.game_state[nx][ny] == player and found_opponent:
                    return True
                else:
                    break
                nx += dx
                ny += dy
        return False

    def make_move(self, game_state, x, y, player, opponent):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        game_state[x][y] = player
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            tiles_to_flip = []
            while 0 <= nx < 8 and 0 <= ny < 8:
                if game_state[nx][ny] == opponent:
                    tiles_to_flip.append((nx, ny))
                elif game_state[nx][ny] == player:
                    for px, py in tiles_to_flip:
                        game_state[px][py] = player
                    break
                else:
                    break
                nx += dx
                ny += dy

    def visualize_board(self, game_state, score, node_id, alpha=None, beta=None, move_label=None):
        if not self.visualize_enabled:
            return
        
        transposed_board = np.transpose(game_state)
        board_str = '\n'.join([' '.join(['_' if cell is None else cell[0] for cell in row]) for row in transposed_board])
        
        alpha_beta_str = f"Alpha: {alpha}, Beta: {beta}\n" if alpha is not None and beta is not None else ""
        
        if move_label:
            label = f"{alpha_beta_str}Score: {score}\n{board_str}\n**{move_label}**"
            color = "lightblue"
        else:
            label = f"{alpha_beta_str}Score: {score}\n{board_str}"
            color = "white"
        
        self.dot.node(node_id, label=label, style='filled', fillcolor=color)

    def get_best_move(self, is_maximizing_player=False):
        best_score = -math.inf if is_maximizing_player else math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        valid_moves = [(x, y) for x in range(8) for y in range(8) if self.is_valid_move(x, y, self.current_player, self.opponent)]
        
        root_node_id = "root"
        
        

        for x, y in valid_moves:
            temp_game_state = np.copy(self.game_state)
            self.make_move(temp_game_state, x, y, self.current_player, self.opponent)
            parent_node_id = root_node_id + f"_{x+1}_{y+1}"
            
            if self.visualize_enabled:
                self.dot.edge(root_node_id, parent_node_id)

            score = self.minimax(temp_game_state, self.depth - 1, alpha, beta, not is_maximizing_player, parent_node_id)
            
            if self.visualize_enabled:
                self.visualize_board(temp_game_state, score, parent_node_id, alpha=alpha, beta=beta,
                                    move_label=f"MOVED: {x+1},{y+1} {self.current_player} {'max' if is_maximizing_player else 'min'}")
            
            print(f"Move {x+1},{y+1} -> Score: {score}") if best_move is not None else print("None moves")

            if is_maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = (x, y)
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (x, y)
                beta = min(beta, score)

            if beta <= alpha:
                break

        if self.visualize_enabled:
            self.visualize_board(self.game_state, best_score, root_node_id, move_label=f"{'min' if is_maximizing_player else 'max'}")
        if self.visualize_enabled:
            self.dot.node(f"root_best", label=f"Chosen Move\n{best_move[0]+1},{best_move[1]+1}", style='filled', fillcolor='green')
            self.dot.edge(root_node_id, f"root_best")
            self.dot.render('othello_tree', format='png', cleanup=True)
        
        return best_move

    def minimax(self, game_state, depth, alpha, beta, is_maximizing, parent_node_id):
        
        if depth == 0 or self.is_game_over():
            score = self.evaluate_board(game_state)
            if self.visualize_enabled:
                self.dot.node(parent_node_id, label=f"Leaf\nScore: {score}", style='filled', fillcolor='lightblue')
            return score
        
        if is_maximizing:
            max_eval = -math.inf
            
            valid_moves = [(x,y) for x in range(8) for y in range(8) if self.is_valid_move(x,y,self.current_player,self.opponent)]
            
            for x,y in valid_moves:
                temp_game_state = np.copy(game_state)
                self.make_move(temp_game_state,x,y,self.current_player, self.opponent)
                
                node_id = f"{parent_node_id}_max_{x+1}_{y+1}"
                
                eval = self.minimax(temp_game_state ,depth - 1 ,alpha ,beta ,False ,node_id)
                
                
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if self.visualize_enabled:
                    self.visualize_board(temp_game_state, max_eval, node_id, alpha=alpha, beta=beta,
                                         move_label=f"MOVED: {x+1},{y+1} {self.current_player} {"min" if is_maximizing else "max"}")
                    
                self.dot.edge(parent_node_id,node_id)
                
                if beta <= alpha:
                    pruned_node_id = f"{node_id}_pruned"
                    if self.visualize_enabled:
                        self.dot.node(pruned_node_id,label=f"Pruned\nAlpha: {alpha}, Beta: {beta}", style='filled', fillcolor='lightcoral')
                        self.dot.edge(parent_node_id ,pruned_node_id)
                    break
            
            return max_eval
        
        else:
            min_eval = math.inf
            
            valid_moves = [(x,y) for x in range(8) for y in range(8) if self.is_valid_move(x,y,self.opponent,self.current_player)]
            
            for x,y in valid_moves:
                temp_game_state = np.copy(game_state)
                self.make_move(temp_game_state,x,y,self.opponent, self.current_player)
                
                node_id = f"{parent_node_id}_min_{x+1}_{y+1}"
                
                eval = self.minimax(temp_game_state ,depth - 1 ,alpha ,beta ,True ,node_id)
                
                
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if self.visualize_enabled:
                    self.visualize_board(temp_game_state,min_eval,node_id ,
                                         alpha=alpha ,
                                         beta=beta ,
                                         move_label=f"MOVED: {x+1},{y+1} {self.opponent} {"max" if not is_maximizing else "min"}")
                    
                self.dot.edge(parent_node_id,node_id)
                
                if beta <= alpha:
                    pruned_node_id = f"{node_id}_pruned"
                    if self.visualize_enabled:
                        self.dot.node(pruned_node_id,label=f"Pruned\nAlpha: {alpha}, Beta: {beta}", style='filled', fillcolor='lightcoral')
                        self.dot.edge(parent_node_id ,pruned_node_id)
                    break
            
            return min_eval

    def evaluate_board(self, game_state):
        score = 0
        for x in range(8):
            for y in range(8):
                if game_state[x][y] == self.current_player:
                    score += self.corner_weights[x][y]
                elif game_state[x][y] == self.opponent:
                    score -= self.corner_weights[x][y]
        return score

    def is_game_over(self):
        current_player_valid = any(
            self.is_valid_move(x, y, self.current_player, self.opponent)
            for x in range(8) for y in range(8)
        )
        opponent_valid = any(
            self.is_valid_move(x, y, self.opponent, self.current_player)
            for x in range(8) for y in range(8)
        )
        return not (current_player_valid or opponent_valid)