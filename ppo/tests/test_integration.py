import gymnasium as gym
import numpy as np
import torch

from config import PPOConfig
from network import DiscreteActorCritic
from ppo import PPO
from rollout_buffer import RolloutBuffer
from evaluate import evaluate


class TestCartPoleSmoke:
    def test_one_update(self):
        """Smoke test: one full rollout + PPO update on CartPole."""
        config = PPOConfig(
            env_name="CartPole-v1",
            num_envs=2,
            rollout_steps=32,
            num_epochs=2,
            num_minibatches=2,
            entropy_coef=0.01,
        )

        envs = gym.vector.SyncVectorEnv([
            lambda: gym.make("CartPole-v1") for _ in range(config.num_envs)
        ])
        obs_dim = envs.single_observation_space.shape[0]
        act_dim = 1

        ac = DiscreteActorCritic(obs_dim, envs.single_action_space.n, config.hidden_dim)
        ppo = PPO(ac, config)
        buffer = RolloutBuffer(config.rollout_steps, config.num_envs, obs_dim, act_dim)

        obs, _ = envs.reset()
        ac.eval()
        for step in range(config.rollout_steps):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, _, value = ac.get_action_and_value(obs_t)

            next_obs, reward, terminated, truncated, _ = envs.step(action.numpy())
            buffer.add(obs, action.numpy().reshape(-1, 1), log_prob.numpy(),
                       reward, terminated.astype(np.float32), value.numpy())
            obs = next_obs

        with torch.no_grad():
            last_value = ac.critic(torch.tensor(obs, dtype=torch.float32)).squeeze(-1).numpy()
        buffer.compute_advantages(last_value, terminated.astype(np.float32),
                                  config.gamma, config.gae_lambda)

        ac.train()
        metrics = ppo.update(buffer)
        envs.close()

        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert metrics["entropy"] > 0

    def test_evaluate_runs(self):
        """Evaluate function runs without error on CartPole."""
        config = PPOConfig(env_name="CartPole-v1", eval_episodes=2)
        ac = DiscreteActorCritic(4, 2)
        returns = evaluate(ac, config, num_episodes=2, discrete=True)
        assert len(returns) == 2
        assert all(r >= 0 for r in returns)
