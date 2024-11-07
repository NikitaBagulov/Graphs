import math

class OthelloAI:
    def __init__(self, game_state, current_player):
        self.game_state = game_state
        self.current_player = current_player
        self.opponent = 'black' if current_player == 'white' else 'white'
        self.depth = 1

    def is_valid_move(self, x, y):
        """Проверяет, является ли ход на (x, y) допустимым для текущего игрока."""
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
        """Делает ход на (x, y) для указанного игрока и переворачивает фишки."""
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
        """Проверяет, завершена ли игра (нет допустимых ходов для обоих игроков)."""
        for player in [self.current_player, self.opponent]:
            for x in range(8):
                for y in range(8):
                    if self.is_valid_move_for_player(game_state, x, y, player):
                        return False
        return True

    def is_valid_move_for_player(self, game_state, x, y, player):
        """Проверяет, является ли ход на (x, y) допустимым для указанного игрока."""
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

    def get_best_move(self):
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        for x in range(8):
            for y in range(8):
                if self.is_valid_move(x, y):
                    temp_game_state = [row[:] for row in self.game_state]
                    self.make_move(temp_game_state, x, y, self.current_player)
                    score = self.minimax(temp_game_state, self.depth - 1, alpha, beta, False)
                    
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)
                    
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break

        return best_move

    def minimax(self, game_state, depth, alpha, beta, is_maximizing):
        if depth == 0 or self.is_game_over(game_state):
            return self.evaluate_board(game_state)

        if is_maximizing:
            max_eval = -math.inf
            for x in range(8):
                for y in range(8):
                    if self.is_valid_move(x, y):
                        temp_game_state = [row[:] for row in game_state]
                        self.make_move(temp_game_state, x, y, self.current_player)
                        eval = self.minimax(temp_game_state, depth - 1, alpha, beta, False)
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = math.inf
            for x in range(8):
                for y in range(8):
                    if self.is_valid_move(x, y):
                        temp_game_state = [row[:] for row in game_state]
                        self.make_move(temp_game_state, x, y, self.current_player)
                        eval = self.minimax(temp_game_state, depth - 1, alpha, beta, True)
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval

    def evaluate_board(self, game_state):
        score = 0
        for row in game_state:
            for cell in row:
                if cell == self.current_player:
                    score += 1
                elif cell is not None:
                    score -= 1
        return score
