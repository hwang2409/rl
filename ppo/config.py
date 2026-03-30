from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Environment
    env_name: str = "HalfCheetah-v5"
    num_envs: int = 8
    normalize_obs: bool = False

    # Rollout
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO
    num_epochs: int = 10
    num_minibatches: int = 32
    clip_epsilon: float = 0.2
    clip_value: bool = True
    max_grad_norm: float = 0.5

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.0

    # Optimization
    learning_rate: float = 3e-4
    anneal_lr: bool = True

    # Training
    total_timesteps: int = 1_000_000

    # Network
    hidden_dim: int = 64

    # Evaluation
    eval_interval: int = 10_000
    eval_episodes: int = 10
    record_video_interval: int = 50_000

    # General
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"


def cartpole_config() -> PPOConfig:
    return PPOConfig(
        env_name="CartPole-v1",
        num_envs=4,
        rollout_steps=128,
        num_epochs=4,
        num_minibatches=4,
        learning_rate=2.5e-4,
        entropy_coef=0.01,
        total_timesteps=100_000,
        eval_interval=5_000,
    )


def pendulum_config() -> PPOConfig:
    return PPOConfig(
        env_name="Pendulum-v1",
        num_envs=4,
        rollout_steps=2048,
        total_timesteps=200_000,
    )


def halfcheetah_config() -> PPOConfig:
    return PPOConfig(
        env_name="HalfCheetah-v5",
        num_envs=8,
        rollout_steps=2048,
        total_timesteps=1_000_000,
        normalize_obs=True,
    )


def ant_config() -> PPOConfig:
    return PPOConfig(
        env_name="Ant-v5",
        num_envs=16,
        rollout_steps=2048,
        total_timesteps=2_000_000,
        normalize_obs=True,
    )
