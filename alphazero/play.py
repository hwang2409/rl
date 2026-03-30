import sys

import numpy as np
import torch

from config import Config
from game import Connect4
from model import AlphaZeroNet
from mcts import search
from utils import get_device


COLORS = {
    "X": "\033[91mX\033[0m",  # Red
    "O": "\033[93mO\033[0m",  # Yellow
    ".": "\033[90m.\033[0m",  # Gray
}


def display_board(game: Connect4) -> None:
    cols_header = "  " + "   ".join(str(i + 1) for i in range(game.cols))
    print(cols_header)
    for r in range(game.rows - 1, -1, -1):
        cells = []
        for c in range(game.cols):
            if game.board[r, c] == 1:
                cells.append(COLORS["X"])
            elif game.board[r, c] == -1:
                cells.append(COLORS["O"])
            else:
                cells.append(COLORS["."])
        print("| " + "   ".join(cells) + " |")
    print("  " + "---" * game.cols + "-")


def get_human_move(game: Connect4) -> int:
    valid = game.get_valid_moves()
    while True:
        try:
            col = int(input(f"\nYour turn. Enter column (1-{game.cols}): ")) - 1
            if 0 <= col < game.cols and valid[col]:
                return col
            print("Invalid move. Try again.")
        except (ValueError, EOFError):
            print("Enter a number.")


def play(checkpoint_path: str, human_first: bool = True):
    config = Config()
    device = get_device(config.device)

    model = AlphaZeroNet(
        rows=config.rows, cols=config.cols,
        num_res_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    iteration = checkpoint.get("iteration", "?")
    print(f"\n{'='*40}")
    print(f"  CONNECT 4 — AlphaZero (iter {iteration})")
    print(f"{'='*40}")
    print(f"  You are {'X (first)' if human_first else 'O (second)'}")
    print(f"  AI uses {config.eval_simulations} MCTS simulations")
    print()

    game = Connect4(config.rows, config.cols, config.win_length)
    human_player = 1 if human_first else -1

    while True:
        display_board(game)
        done, reward = game.is_terminal()
        if done:
            if reward == 0.0:
                print("\nDraw!")
            else:
                last_mover = -game.current_player
                if last_mover == human_player:
                    print("\nYou win!")
                else:
                    print("\nAI wins!")
            break

        if game.current_player == human_player:
            action = get_human_move(game)
        else:
            print("\nAI is thinking...")
            probs, value = search(
                game, model,
                num_simulations=config.eval_simulations,
                c_puct=config.c_puct,
                add_noise=False,
                device=device,
            )
            action = int(np.argmax(probs))
            confidence = probs[action] * 100
            # Value from AI's perspective
            print(f"AI plays column {action + 1} (confidence: {confidence:.0f}%, "
                  f"eval: {value:+.2f})")

        game = game.make_move(action)

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play.py <checkpoint_path> [--second]")
        print("Example: python play.py checkpoints/iter_020.pt")
        sys.exit(1)

    cp = sys.argv[1]
    first = "--second" not in sys.argv
    play(cp, human_first=first)
