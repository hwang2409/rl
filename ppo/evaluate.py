import os

import gymnasium as gym
import numpy as np
import torch

from config import PPOConfig


def evaluate(actor_critic, config: PPOConfig, device: str = "cpu",
             num_episodes: int | None = None, discrete: bool = False) -> list[float]:
    """Run deterministic evaluation episodes.

    Uses the policy mean (no sampling noise) for a fair measure of learned behavior.
    """
    if num_episodes is None:
        num_episodes = config.eval_episodes

    env = gym.make(config.env_name)
    returns = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if discrete:
                    dist, _ = actor_critic(obs_t)
                    action = dist.probs.argmax(dim=-1).cpu().numpy()
                else:
                    action = actor_critic.actor_mean(obs_t).cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action if not discrete else action[0])
            total_reward += reward
            done = terminated or truncated

        returns.append(float(total_reward))

    env.close()
    return returns


def record_video(actor_critic, config: PPOConfig, device: str = "cpu",
                 global_step: int = 0, discrete: bool = False):
    """Record a video of the agent's behavior."""
    video_dir = os.path.join(config.checkpoint_dir, "videos")
    env = gym.make(config.env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_dir,
        name_prefix=f"step_{global_step:07d}",
        episode_trigger=lambda _: True,
    )

    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if discrete:
                dist, _ = actor_critic(obs_t)
                action = dist.probs.argmax(dim=-1).cpu().numpy()
            else:
                action = actor_critic.actor_mean(obs_t).cpu().numpy()[0]
        obs, _, terminated, truncated, _ = env.step(action if not discrete else action[0])
        done = terminated or truncated
    env.close()
