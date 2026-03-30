# Reinforcement Learning from Scratch

Two RL algorithms implemented from scratch in PyTorch, built for learning.

## Projects

### [AlphaZero](alphazero/) — Learn a board game through self-play

Neural network + Monte Carlo Tree Search learns Connect 4 with zero human knowledge. Trains in ~15 min on GPU.

```bash
cd alphazero
python train.py
python play.py checkpoints/iter_020.pt  # play against it
```

53 tests | 376K params | Beats minimax after ~12 iterations

**Concepts:** MCTS, policy/value networks, self-play, exploration vs exploitation

### [PPO](ppo/) — Teach a creature to walk

Proximal Policy Optimization trains simulated robots (CartPole, HalfCheetah, Ant) to move using continuous control. The algorithm behind RLHF/ChatGPT training.

```bash
cd ppo
python train.py cartpole      # sanity check (~30s)
python train.py halfcheetah   # main target (~15 min on GPU)
```

25 tests | ~9K params | CartPole → Pendulum → HalfCheetah → Ant progression

**Concepts:** Policy gradients, GAE, clipped objective, continuous action spaces, vectorized environments

## What's different between them

| | AlphaZero | PPO |
|---|---|---|
| Action space | Discrete (7 columns) | Continuous (joint torques) |
| Planning | MCTS search at each step | Direct policy, no search |
| Learning signal | Self-play game outcomes | Trial-and-error rewards |
| Environment model | Known (game rules) | Unknown (learned from interaction) |
| Network | CNN (spatial board) | MLP (observation vector) |
