import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _init_layer(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic for continuous control.

    Actor outputs a Gaussian: learned mean + state-independent log_std.
    Critic outputs an unbounded scalar V(s).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor_mean = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.actor_log_std.clamp(-20, 2).exp()
        return Normal(mean, std), self.critic(obs)

    def get_action_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action. Returns (action, log_prob, entropy, value)."""
        dist, value = self.forward(obs)
        action = dist.sample()
        # Sum log_prob over action dimensions (independent Gaussian per dimension)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under current policy. Returns (log_prob, entropy, value)."""
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)


class DiscreteActorCritic(nn.Module):
    """Separate actor and critic for discrete action spaces (e.g., CartPole)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs)
        return Categorical(logits=logits), self.critic(obs)

    def get_action_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        return dist.log_prob(actions.squeeze(-1)), dist.entropy(), value.squeeze(-1)
