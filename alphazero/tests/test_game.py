import numpy as np
import pytest

from game import Connect4


class TestInitialization:
    def test_empty_board(self):
        g = Connect4()
        assert g.board.shape == (6, 7)
        assert np.all(g.board == 0)

    def test_all_columns_valid(self):
        g = Connect4()
        assert np.all(g.get_valid_moves())

    def test_player_1_starts(self):
        g = Connect4()
        assert g.current_player == 1


class TestMakeMove:
    def test_piece_falls_to_bottom(self):
        g = Connect4().make_move(3)
        assert g.board[0, 3] == 1

    def test_pieces_stack(self):
        g = Connect4().make_move(3).make_move(3)
        assert g.board[0, 3] == 1
        assert g.board[1, 3] == -1

    def test_player_alternates(self):
        g = Connect4()
        assert g.current_player == 1
        g = g.make_move(0)
        assert g.current_player == -1
        g = g.make_move(0)
        assert g.current_player == 1

    def test_move_count(self):
        g = Connect4().make_move(0).make_move(1).make_move(2)
        assert g.move_count == 3

    def test_returns_new_state(self):
        g1 = Connect4()
        g2 = g1.make_move(3)
        assert np.all(g1.board == 0)
        assert g2.board[0, 3] == 1

    def test_invalid_column_raises(self):
        g = Connect4()
        with pytest.raises(ValueError):
            g.make_move(7)
        with pytest.raises(ValueError):
            g.make_move(-1)

    def test_full_column_raises(self):
        g = Connect4()
        for _ in range(6):
            g = g.make_move(0)
        with pytest.raises(ValueError):
            g.make_move(0)

    def test_full_column_not_valid(self):
        g = Connect4()
        for _ in range(6):
            g = g.make_move(0)
        valid = g.get_valid_moves()
        assert not valid[0]
        assert all(valid[1:])


class TestWinDetection:
    def test_horizontal_win(self):
        g = Connect4()
        # Player 1: cols 0,1,2,3 on row 0
        # Player -1: cols 0,1,2 on row 1 (stacking)
        moves = [0, 0, 1, 1, 2, 2, 3]
        for col in moves:
            g = g.make_move(col)
        done, reward = g.is_terminal()
        assert done
        assert reward == 1.0

    def test_vertical_win(self):
        g = Connect4()
        # Player 1: col 0, rows 0-3
        # Player -1: col 1, rows 0-2
        moves = [0, 1, 0, 1, 0, 1, 0]
        for col in moves:
            g = g.make_move(col)
        done, reward = g.is_terminal()
        assert done
        assert reward == 1.0

    def test_diagonal_up_win(self):
        g = Connect4()
        # Build a diagonal for player 1
        # Col 0: X
        # Col 1: O, X
        # Col 2: O, O, X (need filler)
        # Col 3: O, O, O, X (need filler)
        moves = [0, 1, 1, 2, 3, 2, 2, 3, 3, 6, 3]
        for col in moves:
            g = g.make_move(col)
        done, reward = g.is_terminal()
        assert done
        assert reward == 1.0

    def test_diagonal_down_win(self):
        g = Connect4()
        # Mirror of diagonal up
        # Col 3: X
        # Col 2: O, X
        # Col 1: O, O, X
        # Col 0: O, O, O, X
        moves = [3, 2, 2, 1, 0, 1, 1, 0, 0, 6, 0]
        for col in moves:
            g = g.make_move(col)
        done, reward = g.is_terminal()
        assert done
        assert reward == 1.0

    def test_no_win_yet(self):
        g = Connect4().make_move(0).make_move(1).make_move(2)
        done, reward = g.is_terminal()
        assert not done
        assert reward is None

    def test_draw(self):
        g = Connect4(rows=2, cols=2, win_length=3)
        g = g.make_move(0).make_move(1).make_move(1).make_move(0)
        done, reward = g.is_terminal()
        assert done
        assert reward == 0.0

    def test_player_2_wins(self):
        g = Connect4()
        # Player 1 wastes moves, player -1 gets 4 in a row
        moves = [0, 1, 0, 2, 0, 3, 6, 4]
        for col in moves:
            g = g.make_move(col)
        done, reward = g.is_terminal()
        assert done
        assert reward == 1.0  # from perspective of last mover (player -1)


class TestEncode:
    def test_shape(self):
        g = Connect4()
        enc = g.encode()
        assert enc.shape == (3, 6, 7)
        assert enc.dtype == np.float32

    def test_empty_board(self):
        g = Connect4()
        enc = g.encode()
        assert np.all(enc[0] == 0)
        assert np.all(enc[1] == 0)
        assert np.all(enc[2] == 1)  # player 1 to move

    def test_current_player_perspective(self):
        g = Connect4().make_move(3)
        enc = g.encode()
        # Current player is -1. Channel 0 = -1's pieces (none yet on board for -1)
        assert enc[0, 0, 3] == 0  # -1 has no pieces
        assert enc[1, 0, 3] == 1  # player 1's piece is opponent
        assert np.all(enc[2] == 0)  # player -1 to move

    def test_both_players(self):
        g = Connect4().make_move(0).make_move(1)
        enc = g.encode()
        # Current player is 1 again
        assert enc[0, 0, 0] == 1  # player 1's piece in channel 0
        assert enc[1, 0, 1] == 1  # player -1's piece in channel 1
        assert np.all(enc[2] == 1)  # player 1 to move


class TestClone:
    def test_independence(self):
        g1 = Connect4().make_move(3)
        g2 = g1.clone()
        g2 = g2.make_move(4)
        assert g1.board[0, 4] == 0
        assert g2.board[0, 4] == -1

    def test_deep_copy(self):
        g1 = Connect4().make_move(0)
        g2 = g1.clone()
        g2.board[0, 0] = 0  # mutate clone
        assert g1.board[0, 0] == 1  # original unchanged


class TestRepr:
    def test_displays(self):
        g = Connect4().make_move(3)
        s = repr(g)
        assert "X" in s
        assert "." in s
