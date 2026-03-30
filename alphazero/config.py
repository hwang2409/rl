from dataclasses import dataclass, field


@dataclass
class Config:
    # Game
    rows: int = 6
    cols: int = 7
    win_length: int = 4

    # Neural net
    num_res_blocks: int = 5
    num_channels: int = 64

    # MCTS
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 15

    # Training
    num_iterations: int = 40
    games_per_iteration: int = 100
    num_parallel_games: int = 8
    batch_size: int = 128
    train_epochs: int = 4
    learning_rate: float = 0.001
    lr_milestones: list = field(default_factory=lambda: [10, 15])
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    replay_buffer_size: int = 50_000
    min_replay_size: int = 1_000

    # Evaluation
    eval_games: int = 40
    eval_simulations: int = 100

    # Device
    device: str = "cuda"
