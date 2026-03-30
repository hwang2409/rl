from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    """Stores rollout data from vectorized environments.

    Layout: (rollout_steps, num_envs, ...). Flattened to
    (rollout_steps * num_envs, ...) for minibatch generation.
    """

    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int, act_dim: int):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self._allocate()

    def _allocate(self):
        s, e = self.rollout_steps, self.num_envs
        self.observations = np.zeros((s, e, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((s, e, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((s, e), dtype=np.float32)
        self.rewards = np.zeros((s, e), dtype=np.float32)
        self.terminateds = np.zeros((s, e), dtype=np.float32)
        self.values = np.zeros((s, e), dtype=np.float32)
        self.advantages = np.zeros((s, e), dtype=np.float32)
        self.returns = np.zeros((s, e), dtype=np.float32)

    def add(self, obs: np.ndarray, actions: np.ndarray, log_probs: np.ndarray,
            rewards: np.ndarray, terminateds: np.ndarray, values: np.ndarray):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.terminateds[self.ptr] = terminateds
        self.values[self.ptr] = values
        self.ptr += 1

    def compute_advantages(self, last_value: np.ndarray, last_terminated: np.ndarray,
                           gamma: float, gae_lambda: float):
        """Compute GAE(lambda) advantages and discounted returns.

        GAE trades off bias vs variance through lambda:
          lambda=0: one-step TD (high bias, low variance)
          lambda=1: full Monte Carlo (low bias, high variance)
          lambda=0.95: sweet spot for most environments

        Uses `terminated` (not truncated) for value masking — when an episode
        ends naturally the next state has no future value, but when truncated
        by time limit the value should still be bootstrapped.
        """
        last_gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - last_terminated
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.terminateds[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae
            last_gae = self.advantages[t]

        self.returns = self.advantages + self.values

    def get_minibatches(self, num_minibatches: int, device: str = "cpu"):
        """Flatten, shuffle, yield minibatches with normalized advantages."""
        total = self.rollout_steps * self.num_envs
        batch_size = total // num_minibatches
        indices = np.random.permutation(total)

        # Flatten all arrays: (steps, envs, ...) -> (steps*envs, ...)
        flat_obs = self.observations.reshape(total, -1)
        flat_actions = self.actions.reshape(total, -1)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_values = self.values.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]

            adv = flat_advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            yield Batch(
                obs=torch.tensor(flat_obs[idx], device=device),
                actions=torch.tensor(flat_actions[idx], device=device),
                old_log_probs=torch.tensor(flat_log_probs[idx], device=device),
                advantages=torch.tensor(adv, device=device),
                returns=torch.tensor(flat_returns[idx], device=device),
                old_values=torch.tensor(flat_values[idx], device=device),
            )

    def reset(self):
        self.ptr = 0
