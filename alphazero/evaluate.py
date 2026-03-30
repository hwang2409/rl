import math

import numpy as np

from config import Config
from game import Connect4
from model import AlphaZeroNet
from mcts import search


# --- Baseline agents ---

class RandomAgent:
    def select_action(self, game: Connect4) -> int:
        valid = game.get_valid_moves()
        return int(np.random.choice(np.where(valid)[0]))


class OneStepLookahead:
    def select_action(self, game: Connect4) -> int:
        valid = np.where(game.get_valid_moves())[0]

        # Win if possible
        for col in valid:
            next_state = game.make_move(col)
            done, reward = next_state.is_terminal()
            if done and reward == 1.0:
                return int(col)

        # Block opponent win
        for col in valid:
            # Simulate opponent playing here
            hypothetical = game.clone()
            hypothetical.current_player *= -1
            next_state = hypothetical.make_move(col)
            done, reward = next_state.is_terminal()
            if done and reward == 1.0:
                return int(col)

        return int(np.random.choice(valid))


class MinimaxAgent:
    def __init__(self, depth: int = 4):
        self.depth = depth

    def select_action(self, game: Connect4) -> int:
        best_score = -math.inf
        best_action = None
        valid = np.where(game.get_valid_moves())[0]

        for col in valid:
            next_state = game.make_move(col)
            score = -self._minimax(next_state, self.depth - 1, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_action = col

        return int(best_action)

    def _minimax(self, game: Connect4, depth: int, alpha: float, beta: float) -> float:
        """Negamax with alpha-beta pruning. Returns score for current player."""
        done, reward = game.is_terminal()
        if done:
            # reward is from last mover's perspective (opponent of current player)
            return -reward * (depth + 1)  # prefer faster wins

        if depth == 0:
            return self._evaluate(game)

        valid = np.where(game.get_valid_moves())[0]
        best = -math.inf

        for col in valid:
            next_state = game.make_move(col)
            score = -self._minimax(next_state, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best

    def _evaluate(self, game: Connect4) -> float:
        """Heuristic board evaluation for current player."""
        score = 0.0
        board = game.board
        player = game.current_player

        # Center column preference
        center_col = game.cols // 2
        center_count = np.sum(board[:, center_col] == player)
        opp_center = np.sum(board[:, center_col] == -player)
        score += (center_count - opp_center) * 3

        # Check all windows of 4
        for r in range(game.rows):
            for c in range(game.cols):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    window = []
                    for i in range(game.win_length):
                        nr, nc = r + i * dr, c + i * dc
                        if 0 <= nr < game.rows and 0 <= nc < game.cols:
                            window.append(board[nr, nc])
                        else:
                            break
                    if len(window) == game.win_length:
                        score += self._score_window(window, player)

        return score / 100.0  # normalize

    @staticmethod
    def _score_window(window: list, player: int) -> float:
        p_count = sum(1 for x in window if x == player)
        o_count = sum(1 for x in window if x == -player)
        empty = sum(1 for x in window if x == 0)

        if p_count == 4:
            return 1000
        if o_count == 4:
            return -1000
        if p_count == 3 and empty == 1:
            return 50
        if o_count == 3 and empty == 1:
            return -50
        if p_count == 2 and empty == 2:
            return 10
        if o_count == 2 and empty == 2:
            return -10
        return 0


class MCTSAgent:
    """Wraps the AlphaZero model + MCTS for evaluation games."""

    def __init__(self, model: AlphaZeroNet, config: Config, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device

    def select_action(self, game: Connect4) -> int:
        probs, _ = search(
            game, self.model,
            num_simulations=self.config.eval_simulations,
            c_puct=self.config.c_puct,
            add_noise=False,
            device=self.device,
        )
        return int(np.argmax(probs))


# --- Evaluation ---

def play_evaluation_game(agent1, agent2) -> int:
    """Play a game between two agents. Returns 1 if agent1 wins, -1 if agent2 wins, 0 for draw."""
    game = Connect4()
    agents = {1: agent1, -1: agent2}

    while True:
        done, reward = game.is_terminal()
        if done:
            if reward == 0.0:
                return 0
            # Last mover won. Last mover = -game.current_player
            last_mover = -game.current_player
            return 1 if last_mover == 1 else -1

        action = agents[game.current_player].select_action(game)
        game = game.make_move(action)


def evaluate_against_baselines(model: AlphaZeroNet, config: Config,
                                device: str = "cpu") -> dict:
    """Evaluate the model against all baseline agents."""
    ai = MCTSAgent(model, config, device=device)
    baselines = {
        "Random": RandomAgent(),
        "Lookahead": OneStepLookahead(),
        "Minimax(4)": MinimaxAgent(depth=4),
    }

    results = {}
    for name, opponent in baselines.items():
        wins, losses, draws = 0, 0, 0
        half = config.eval_games // 2

        for game_num in range(config.eval_games):
            if game_num < half:
                result = play_evaluation_game(ai, opponent)
            else:
                result = -play_evaluation_game(opponent, ai)

            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        win_rate = (wins + 0.5 * draws) / config.eval_games * 100
        results[name] = {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}
        print(f"  vs {name:>12s}: W{wins} L{losses} D{draws} ({win_rate:.1f}%)")

    return results


def compute_elo(win_rate: float, opponent_elo: float) -> float:
    """Estimate ELO from win rate against a known-ELO opponent."""
    if win_rate >= 1.0:
        win_rate = 0.99
    elif win_rate <= 0.0:
        win_rate = 0.01
    return opponent_elo - 400 * math.log10(1 / win_rate - 1)
