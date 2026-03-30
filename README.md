# AlphaZero-Lite

A from-scratch implementation of [AlphaZero](https://arxiv.org/abs/1712.01815) that learns to play Connect 4 through pure self-play. No human game data, no hand-crafted heuristics — just a neural network, Monte Carlo Tree Search, and millions of games against itself.

## How it works

AlphaZero combines three ideas:

1. **Neural network** — takes a board position, outputs a policy (which moves look good) and a value (who's winning). Starts random, improves through training.

2. **Monte Carlo Tree Search (MCTS)** — before each move, simulates hundreds of future game trajectories using the neural net to guide the search. Balances exploring new moves vs. going deeper on promising ones.

3. **Self-play loop** — the agent plays games against itself, records every position + what MCTS recommended + who won, then trains the neural net on this data. Better net → better search → better training data → repeat.

```
┌─────────────────────────────────────────────┐
│              TRAINING ITERATION              │
│                                              │
│   Self-play ──→ Replay buffer ──→ Training   │
│       │                              │       │
│       └──────────────────────────────┘       │
│            neural net improves               │
└─────────────────────────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt  # torch, numpy, pytest
```

## Train

```bash
python train.py
```

Default config trains for 20 iterations (~15 min on a GPU). Checkpoints are saved to `checkpoints/` after each iteration. Training logs show loss curves and win rates against baseline agents:

```
--- Iteration 5/20 ---
Generating 50 self-play games...
  Self-play: 24.3s | Games: 50 | Examples: 2,146 | Buffer: 10,730
  Train loss: 1.834 (policy: 1.421, value: 0.413) | LR: 0.001000
  Evaluating...
  vs       Random: W40 L0 D0 (100.0%)
  vs    Lookahead: W35 L3 D2 (90.0%)
  vs   Minimax(4): W12 L22 D6 (37.5%)
```

Edit `config.py` to tune hyperparameters. For a stronger agent:

```python
Config(num_iterations=40, games_per_iteration=100, num_simulations=200)
```

## Play

```bash
python play.py checkpoints/iter_020.pt           # you go first
python play.py checkpoints/iter_020.pt --second   # AI goes first
```

```
  1   2   3   4   5   6   7
| .   .   .   .   .   .   . |
| .   .   .   .   .   .   . |
| .   .   .   .   .   .   . |
| .   .   X   .   .   .   . |
| .   .   O   X   .   .   . |
| .   X   O   O   X   .   . |
  ---------------------

AI plays column 4 (confidence: 87%, eval: +0.34)
```

## Test

```bash
python -m pytest tests/ -v
```

53 tests covering game logic, neural net shapes, MCTS correctness (forced wins/blocks), and an end-to-end mini training loop.

## Architecture

```
Input (3, 6, 7) → Conv 3→64 + BN + ReLU
  → 5× ResBlock (Conv+BN+ReLU+Conv+BN + skip)
  ├─→ Policy head → softmax over 7 columns
  └─→ Value head → tanh → [-1, 1]
```

376K parameters. The 3 input channels encode: current player's pieces, opponent's pieces, and a color plane (whose turn it is). The board is always encoded from the current player's perspective so a single network plays both sides.

## Project structure

```
config.py        Hyperparameters (dataclass)
game.py          Connect 4 environment
model.py         ResNet with policy + value heads
mcts.py          Monte Carlo Tree Search
self_play.py     Self-play data generation
train.py         Training loop
evaluate.py      Baseline agents + evaluation
play.py          Human vs AI terminal UI
utils.py         Seed, device, timer helpers
tests/           53 tests
```

## Key concepts for learning RL

| Concept | Where to find it |
|---------|-----------------|
| Policy learning | `model.py` — policy head outputs move probabilities |
| Value estimation | `model.py` — value head estimates who's winning |
| Exploration vs exploitation | `mcts.py` — UCB formula balances both |
| Self-play / curriculum learning | `self_play.py` — agent generates its own training data |
| Temporal credit assignment | `self_play.py` — game outcome propagated back to every move |
| Function approximation | `model.py` — neural net replaces tabular value functions |

## References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) — the original AlphaZero paper
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) — AlphaGo Zero, the predecessor
