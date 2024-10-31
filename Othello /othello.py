import pygame
import sys

# Константы
WIDTH, HEIGHT = 400, 400
CELL_SIZE = WIDTH // 8

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)

class OthelloGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Othello Game")
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = 'white'
        self.initialize_pieces()
        self.game_over = False

    def initialize_pieces(self):
        # Установка начальных дисков
        self.board[3][3] = 'white'
        self.board[4][4] = 'white'
        self.board[3][4] = 'black'
        self.board[4][3] = 'black'

    def draw_board(self):
        # Отрисовка доски
        self.screen.fill(GREEN)
        for i in range(8):
            for j in range(8):
                pygame.draw.rect(self.screen, BLACK if (i + j) % 2 == 0 else GREEN,
                                 (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)
                if self.board[i][j] == 'black':
                    pygame.draw.circle(self.screen, BLACK,
                                       (i * CELL_SIZE + CELL_SIZE // 2, j * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)
                elif self.board[i][j] == 'white':
                    pygame.draw.circle(self.screen, WHITE,
                                       (i * CELL_SIZE + CELL_SIZE // 2, j * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)

    def play_move(self, x, y):
        if not self.game_over and self.is_valid_move(x, y):
            self.make_move(x, y)
            if not any(self.is_valid_move(i, j) for i in range(8) for j in range(8)):
                self.end_game()
            else:
                # Смена игрока и ход AI
                if self.current_player == 'white':
                    self.current_player = 'black'
                    ai_move = self.get_best_move()
                    if ai_move:
                        self.make_move(ai_move[0], ai_move[1])
                        if not any(self.is_valid_move(i, j) for i in range(8) for j in range(8)):
                            self.end_game()
                        else:
                            self.current_player = 'white'
                else:
                    self.current_player = 'white'

    def is_valid_move(self, x, y):
        if self.board[x][y] is not None:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        opponent = 'black' if self.current_player == 'white' else 'white'
        valid = False

        # Проверка во всех направлениях
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False

            # Двигаемся в направлении, пока находим фишки противника
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] is None:
                    break
                if self.board[nx][ny] == opponent:
                    found_opponent = True
                elif self.board[nx][ny] == self.current_player and found_opponent:
                    # Ход допустим, если находим фишку игрока после фишек противника
                    valid = True
                    break
                else:
                    break
                nx += dx
                ny += dy

        return valid

    def make_move(self, x, y):
        # Устанавливаем фишку текущего игрока на доску
        self.board[x][y] = self.current_player

        # Определяем направления для поиска фишек, которые можно захватить
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Проверяем в каждом направлении
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            to_flip = []

            # Проверяем, что следующие фишки принадлежат противнику
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] is None:
                    break
                if self.board[nx][ny] == self.current_player:
                    # Когда нашли фишку текущего игрока, все фишки в `to_flip` подлежат захвату
                    for flip_x, flip_y in to_flip:
                        self.board[flip_x][flip_y] = self.current_player
                    break
                else:
                    # Добавляем фишку противника для захвата
                    to_flip.append((nx, ny))
                nx += dx
                ny += dy

    def end_game(self):
        # Обработка конца игры и подсчет очков
        black_count = sum(row.count('black') for row in self.board)
        white_count = sum(row.count('white') for row in self.board)
        
        result_text = f"Game Over! Black: {black_count}, White: {white_count}"
        
        print(result_text)
        
        # Окончание игры
        if black_count > white_count:
            winner_text = "Black wins!"
        elif white_count > black_count:
            winner_text = "White wins!"
        else:
            winner_text = "It's a tie!"
        
        print(winner_text)
        
        # Остановка игры
        self.game_over = True

    def evaluate_board(self):
        # Оценка состояния доски для AI
        black_score = sum(row.count('black') for row in self.board)
        white_score = sum(row.count('white') for row in self.board)

        return black_score - white_score

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or not any(self.is_valid_move(i, j) for i in range(8) for j in range(8)):
            return self.evaluate_board()

        if maximizing_player:
            max_eval = float('-inf')
            for i in range(8):
                for j in range(8):
                    if self.is_valid_move(i, j):
                        original_piece_color = self.board[i][j]
                        self.make_move(i, j)
                        eval = self.alpha_beta(depth - 1, alpha, beta, False)
                        self.undo_move(i, j)

                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval

        else:
            min_eval = float('inf')
            for i in range(8):
                for j in range(8):
                    if self.is_valid_move(i, j):
                        original_piece_color = self.board[i][j]
                        self.make_move(i, j)
                        eval = self.alpha_beta(depth - 1, alpha, beta, True)
                        self.undo_move(i, j)

                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval

            
    def get_best_move(self):
         best_score=float('-inf')
         best_move=None
        
         for i in range(8):
             for j in range(8):
                 if(self.is_valid_move(i,j)):
                     # Сделать ход для оценки 
                     print(f"BEST MOVE {i},{j}") 

                     score=self.alpha_beta(3,float('-inf'),float('inf'),False)

                     if score > best_score:
                         best_score=score 
                         best_move=(i,j)

         return best_move

    def undo_move(self, x, y):
        # Очищаем текущий ход
        self.board[x][y] = None

        # Определяем оригинальный цвет фишек, чтобы вернуть его при отмене
        original_piece_color = 'white' if self.current_player == 'black' else 'black'
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Восстанавливаем фишки в каждом направлении
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            to_flip = []

            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] is None or self.board[nx][ny] == self.current_player:
                    break
                to_flip.append((nx, ny))
                nx += dx
                ny += dy

            # Если достигли фишки текущего игрока, переворачиваем все фишки в направлении на оригинальный цвет
            if 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == self.current_player:
                for flip_x, flip_y in to_flip:
                    self.board[flip_x][flip_y] = original_piece_color

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Левый клик мыши
                    mouse_x, mouse_y = event.pos
                    grid_x = mouse_x // CELL_SIZE
                    grid_y = mouse_y // CELL_SIZE
                    self.play_move(grid_x, grid_y)

            # Отрисовка доски и обновление экрана
            self.draw_board()
            pygame.display.flip()

if __name__ == "__main__":
    game = OthelloGame()
    game.run()