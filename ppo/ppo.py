from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import PPOConfig
from rollout_buffer import RolloutBuffer


class PPO:
    """Proximal Policy Optimization with clipped objective.

    Core idea: update the policy to maximize expected returns, but clip the
    probability ratio so the policy never changes too much in one step.

    The clipping creates a trust region:
      - ratio > 1+eps: action became more likely → clip stops further increase
      - ratio < 1-eps: action became less likely → clip stops further decrease
    This prevents destructive updates that collapse the policy.
    """

    def __init__(self, actor_critic: nn.Module, config: PPOConfig):
        self.ac = actor_critic
        self.config = config
        self.optimizer = Adam(actor_critic.parameters(), lr=config.learning_rate, eps=1e-5)

    def update(self, buffer: RolloutBuffer, device: str = "cpu") -> dict[str, float]:
        """Run PPO update on collected rollout data.

        Returns dict of metrics for logging.
        """
        metrics = defaultdict(list)

        for epoch in range(self.config.num_epochs):
            for batch in buffer.get_minibatches(self.config.num_minibatches, device=device):
                # Evaluate old actions under current policy
                new_log_probs, entropy, new_values = self.ac.evaluate_actions(
                    batch.obs, batch.actions
                )

                # Probability ratio: how much the policy changed
                log_ratio = new_log_probs - batch.old_log_probs
                ratio = log_ratio.exp()

                # Approximate KL divergence for monitoring
                # (Schulman's better approximation)
                approx_kl = ((ratio - 1) - log_ratio).mean()

                # Clipped policy objective
                pg_loss1 = -batch.advantages * ratio
                pg_loss2 = -batch.advantages * torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped)
                if self.config.clip_value:
                    v_clipped = batch.old_values + torch.clamp(
                        new_values - batch.old_values,
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon,
                    )
                    v_loss1 = (new_values - batch.returns) ** 2
                    v_loss2 = (v_clipped - batch.returns) ** 2
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = 0.5 * ((new_values - batch.returns) ** 2).mean()

                # Entropy bonus encourages exploration
                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                clip_frac = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_loss.item())
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clip_fraction"].append(clip_frac.item())

        return {k: np.mean(v) for k, v in metrics.items()}
