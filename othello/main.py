import tkinter as tk
from ai import OthelloAI

WIDTH, HEIGHT = 600, 600
CELL_SIZE = WIDTH // 8

class OthelloGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello Game")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="green")
        self.canvas.pack()

        self.score_label = tk.Label(root, text="")
        self.score_label.pack()
        self.restart_button = tk.Button(root, text="Restart", command=self.restart_game)
        self.restart_button.pack()

        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = 'white'
        self.game_over = False

        self.initialize_pieces()
        self.update_score()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.draw_board()

    def initialize_pieces(self):
        self.board[3][3] = 'white'
        self.board[4][4] = 'white'
        self.board[3][4] = 'black'
        self.board[4][3] = 'black'

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(8):
            for j in range(8):
                x1, y1 = i * CELL_SIZE, j * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="green")

                if self.board[i][j] == 'black':
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="black")
                elif self.board[i][j] == 'white':
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="white")

    def handle_click(self, event):
        if self.game_over:
            return

        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if self.is_valid_move(x, y):
            self.make_move(x, y)
            self.switch_player()
            self.draw_board()
            self.update_score()

            if self.is_game_over():
                self.game_over = True
                self.show_winner()

    def is_valid_move(self, x, y):
        if self.board[x][y] is not None:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        opponent = 'black' if self.current_player == 'white' else 'white'
        valid = False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False

            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] is None:
                    break
                if self.board[nx][ny] == opponent:
                    found_opponent = True
                elif self.board[nx][ny] == self.current_player and found_opponent:
                    valid = True
                    break
                else:
                    break
                nx += dx
                ny += dy

        return valid

    def make_move(self, x, y):
        self.board[x][y] = self.current_player

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            to_flip = []

            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] is None:
                    break
                if self.board[nx][ny] == self.current_player:
                    for flip_x, flip_y in to_flip:
                        self.board[flip_x][flip_y] = self.current_player
                    break
                else:
                    to_flip.append((nx, ny))
                nx += dx
                ny += dy

    def switch_player(self):
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        
        if self.current_player == 'black' and not self.game_over:
            self.root.after(1000, self.bot_move)
        else:
            self.draw_board()

    def bot_move(self):
        game_state, current_player = self.get_game_state()
        ai = OthelloAI(game_state, current_player)
        best_move = ai.get_best_move()
        print(f"Лучший ход: ({best_move[0]+1}, {best_move[1]+1})")
        if best_move:
            x, y = best_move
            self.make_move(x, y)
        self.switch_player()

    def get_game_state(self):
        return [row[:] for row in self.board], self.current_player

    def count_scores(self):
        white_score = sum(row.count('white') for row in self.board)
        black_score = sum(row.count('black') for row in self.board)
        return white_score, black_score

    def update_score(self):
        white_score, black_score = self.count_scores()
        self.score_label.config(text=f"White: {white_score} - Black: {black_score}")

    def is_game_over(self):
        return all(self.board[i][j] is not None for i in range(8) for j in range(8)) or \
               not any(self.is_valid_move(x, y) for x in range(8) for y in range(8))

    def show_winner(self):
        white_score, black_score = self.count_scores()
        if white_score > black_score:
            winner = "White wins!"
        elif black_score > white_score:
            winner = "Black wins!"
        else:
            winner = "It's a tie!"
        self.score_label.config(text=f"{winner} Final Score - White: {white_score}, Black: {black_score}")

    def restart_game(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = 'white'
        self.game_over = False
        self.initialize_pieces()
        self.update_score()
        self.draw_board()



if __name__ == "__main__":
    root = tk.Tk()
    game = OthelloGame(root)
    root.mainloop()
