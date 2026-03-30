import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import torch

from config import Config
from game import Connect4
from model import AlphaZeroNet
from mcts import search, select_action


@dataclass
class TrainingExample:
    state: np.ndarray       # (3, rows, cols)
    policy: np.ndarray      # (cols,)
    value: float            # from current player's perspective


def play_game(model: AlphaZeroNet, config: Config, device: str = "cpu") -> list[TrainingExample]:
    """Play a single self-play game and return training examples."""
    game = Connect4(config.rows, config.cols, config.win_length)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    while True:
        terminal, _ = game.is_terminal()
        if terminal:
            break

        temp = 1.0 if game.move_count < config.temperature_threshold else 0.01
        action_probs, _ = search(
            game, model,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            add_noise=True,
            device=device,
        )

        history.append((game.encode(), action_probs, game.current_player))
        action = select_action(action_probs, temperature=temp)
        game = game.make_move(action)

    # Determine game outcome
    _, result = game.is_terminal()
    # result is 1.0 if last mover won, 0.0 for draw
    # last mover is -game.current_player (since current_player flipped after last move)
    last_mover = -game.current_player

    examples = []
    for state, policy, player in history:
        # Value from this position's current player perspective
        if result == 0.0:
            value = 0.0
        elif player == last_mover:
            value = result   # last mover won with result=1.0
        else:
            value = -result  # opponent of last mover lost
        examples.append(TrainingExample(state, policy, value))

    # Data augmentation: horizontal flip
    augmented = []
    for ex in examples:
        augmented.append(ex)
        augmented.append(TrainingExample(
            state=np.flip(ex.state, axis=2).copy(),
            policy=np.flip(ex.policy).copy(),
            value=ex.value,
        ))

    return augmented


def _worker_play_games(args: tuple) -> list[TrainingExample]:
    """Worker function for multiprocessing. Reconstructs model from state_dict."""
    state_dict, config, num_games = args
    torch.set_num_threads(1)  # avoid thread contention between workers

    model = AlphaZeroNet(
        rows=config.rows, cols=config.cols,
        num_res_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    all_examples = []
    for _ in range(num_games):
        all_examples.extend(play_game(model, config, device="cpu"))
    return all_examples


def generate_self_play_data(model: AlphaZeroNet, config: Config,
                            device: str = "cpu") -> list[TrainingExample]:
    """Generate self-play data for one iteration using parallel workers."""
    model.eval()
    state_dict = model.state_dict()

    num_workers = config.num_parallel_games
    # Distribute games evenly across workers
    games_per_worker = [config.games_per_iteration // num_workers] * num_workers
    for i in range(config.games_per_iteration % num_workers):
        games_per_worker[i] += 1

    worker_args = [(state_dict, config, n) for n in games_per_worker]

    with mp.Pool(num_workers) as pool:
        results = pool.map(_worker_play_games, worker_args)

    all_examples = []
    for examples in results:
        all_examples.extend(examples)
    return all_examples
