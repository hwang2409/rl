import numpy as np
import torch

from config import PPOConfig
from network import ActorCritic
from ppo import PPO
from rollout_buffer import RolloutBuffer


def _make_filled_buffer(steps=8, envs=2, obs_dim=4, act_dim=2):
    """Create a buffer with random data and computed advantages."""
    buf = RolloutBuffer(steps, envs, obs_dim, act_dim)
    ac = ActorCritic(obs_dim, act_dim)

    obs = np.random.randn(envs, obs_dim).astype(np.float32)
    for i in range(steps):
        with torch.no_grad():
            obs_t = torch.tensor(obs)
            action, log_prob, _, value = ac.get_action_and_value(obs_t)

        buf.add(obs, action.numpy(), log_prob.numpy(),
                np.random.randn(envs).astype(np.float32),
                np.zeros(envs, dtype=np.float32),
                value.numpy())
        obs = np.random.randn(envs, obs_dim).astype(np.float32)

    with torch.no_grad():
        last_value = ac.critic(torch.tensor(obs)).squeeze(-1).numpy()
    buf.compute_advantages(last_value, np.zeros(envs, dtype=np.float32),
                           gamma=0.99, gae_lambda=0.95)
    return buf, ac


class TestPPOUpdate:
    def test_loss_finite(self):
        config = PPOConfig(num_epochs=2, num_minibatches=2)
        buf, ac = _make_filled_buffer()
        ppo = PPO(ac, config)
        metrics = ppo.update(buf)
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["entropy"])

    def test_metrics_returned(self):
        config = PPOConfig(num_epochs=1, num_minibatches=2)
        buf, ac = _make_filled_buffer()
        ppo = PPO(ac, config)
        metrics = ppo.update(buf)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics

    def test_kl_starts_near_zero(self):
        """First update should have small KL since policy hasn't changed much."""
        config = PPOConfig(num_epochs=1, num_minibatches=2)
        buf, ac = _make_filled_buffer()
        ppo = PPO(ac, config)
        metrics = ppo.update(buf)
        assert metrics["approx_kl"] < 1.0

    def test_multiple_updates_dont_crash(self):
        config = PPOConfig(num_epochs=3, num_minibatches=4)
        buf, ac = _make_filled_buffer(steps=16, envs=4)
        ppo = PPO(ac, config)
        for _ in range(3):
            buf2, _ = _make_filled_buffer(steps=16, envs=4)
            # Use the same ac but fresh buffer data
            metrics = ppo.update(buf2)
            assert np.isfinite(metrics["policy_loss"])


class TestClipping:
    def test_clip_fraction_bounded(self):
        config = PPOConfig(num_epochs=2, num_minibatches=2, clip_epsilon=0.2)
        buf, ac = _make_filled_buffer()
        ppo = PPO(ac, config)
        metrics = ppo.update(buf)
        assert 0 <= metrics["clip_fraction"] <= 1.0

    def test_no_value_clipping(self):
        config = PPOConfig(num_epochs=1, num_minibatches=2, clip_value=False)
        buf, ac = _make_filled_buffer()
        ppo = PPO(ac, config)
        metrics = ppo.update(buf)
        assert np.isfinite(metrics["value_loss"])
