import numpy as np


class Connect4:
    """Connect 4 game environment.

    Board uses row 0 = bottom, row 5 = top. Players are 1 and -1.
    """

    def __init__(self, rows: int = 6, cols: int = 7, win_length: int = 4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.heights = np.zeros(cols, dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.last_move: int | None = None
        self._winner: int | None = None

    def clone(self) -> "Connect4":
        g = Connect4.__new__(Connect4)
        g.rows = self.rows
        g.cols = self.cols
        g.win_length = self.win_length
        g.board = self.board.copy()
        g.heights = self.heights.copy()
        g.current_player = self.current_player
        g.move_count = self.move_count
        g.last_move = self.last_move
        g._winner = self._winner
        return g

    def get_valid_moves(self) -> np.ndarray:
        return self.heights < self.rows

    def make_move(self, col: int) -> "Connect4":
        """Play a piece in the given column. Returns a new game state."""
        if not (0 <= col < self.cols):
            raise ValueError(f"Column {col} out of range")
        if self.heights[col] >= self.rows:
            raise ValueError(f"Column {col} is full")

        new = self.clone()
        row = new.heights[col]
        new.board[row, col] = new.current_player
        new.heights[col] += 1
        new.last_move = col
        new.move_count += 1

        if new._check_win(row, col):
            new._winner = new.current_player

        new.current_player *= -1
        return new

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the piece at (row, col) completes a winning line."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Check both directions along this axis
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while (
                    0 <= r < self.rows
                    and 0 <= c < self.cols
                    and self.board[r, c] == player
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.win_length:
                return True
        return False

    def is_terminal(self) -> tuple[bool, float | None]:
        """Returns (is_done, reward).

        Reward is from the perspective of the player who just moved:
        1.0 = last mover won, -1.0 = last mover lost, 0.0 = draw.
        """
        if self._winner is not None:
            # Winner is the player who just placed (before current_player flipped)
            # Since current_player was flipped after the move, winner = -current_player
            return True, 1.0

        if self.move_count == self.rows * self.cols:
            return True, 0.0

        return False, None

    def encode(self) -> np.ndarray:
        """Encode board from current player's perspective.

        Returns shape (3, rows, cols) float32:
          Channel 0: current player's pieces
          Channel 1: opponent's pieces
          Channel 2: color plane (1 if player 1 to move, 0 otherwise)
        """
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        if self.current_player == 1:
            state[2] = 1.0
        return state

    def __repr__(self) -> str:
        cols_header = "  " + " ".join(str(i + 1) for i in range(self.cols))
        rows_str = []
        for r in range(self.rows - 1, -1, -1):
            cells = []
            for c in range(self.cols):
                if self.board[r, c] == 1:
                    cells.append("X")
                elif self.board[r, c] == -1:
                    cells.append("O")
                else:
                    cells.append(".")
            rows_str.append("| " + " ".join(cells) + " |")

        divider = "  " + "-" * (self.cols * 2 - 1)
        return cols_header + "\n" + "\n".join(rows_str) + "\n" + divider
