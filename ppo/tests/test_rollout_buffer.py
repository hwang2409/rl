import numpy as np
import pytest

from rollout_buffer import RolloutBuffer


class TestGAE:
    def test_single_step_no_terminal(self):
        """A = r + gamma*V_next - V (single step, not terminal)."""
        buf = RolloutBuffer(1, 1, 2, 1)
        buf.add(
            obs=np.zeros((1, 2)),
            actions=np.zeros((1, 1)),
            log_probs=np.zeros(1),
            rewards=np.array([1.0]),
            terminateds=np.array([0.0]),
            values=np.array([0.5]),
        )
        last_value = np.array([0.8])
        last_terminated = np.array([0.0])
        buf.compute_advantages(last_value, last_terminated, gamma=0.99, gae_lambda=0.95)

        expected_delta = 1.0 + 0.99 * 0.8 - 0.5  # = 1.292
        assert abs(buf.advantages[0, 0] - expected_delta) < 1e-5

    def test_terminal_zeroes_bootstrap(self):
        """When episode terminates, next value should be zero."""
        buf = RolloutBuffer(2, 1, 2, 1)
        # Step 0: not terminal (terminateds[0]=0), reward=1.0, value=0.5
        buf.add(np.zeros((1, 2)), np.zeros((1, 1)), np.zeros(1),
                np.array([1.0]), np.array([0.0]), np.array([0.5]))
        # Step 1: terminal (terminateds[1]=1), reward=2.0, value=0.3
        buf.add(np.zeros((1, 2)), np.zeros((1, 1)), np.zeros(1),
                np.array([2.0]), np.array([1.0]), np.array([0.3]))

        # last_terminated matches terminateds[-1] as in real training
        last_value = np.array([0.0])
        last_terminated = np.array([1.0])
        buf.compute_advantages(last_value, last_terminated, gamma=0.99, gae_lambda=0.95)

        # t=1 (last step): terminateds[1]=1, so last_terminated=1
        # next_non_terminal = 1 - 1 = 0, zeroes out bootstrap
        # delta_1 = 2.0 + 0.99*0.0*0 - 0.3 = 1.7
        delta_1 = 2.0 - 0.3
        adv_1 = delta_1

        # t=0: terminateds[0]=0, action was not terminal, next state valid
        # next_non_terminal = 1 - 0 = 1, bootstraps from values[1]=0.3
        # delta_0 = 1.0 + 0.99*0.3*1 - 0.5 = 0.797
        delta_0 = 1.0 + 0.99 * 0.3 - 0.5
        adv_0 = delta_0 + 0.99 * 0.95 * 1.0 * adv_1

        assert abs(buf.advantages[1, 0] - adv_1) < 1e-5
        assert abs(buf.advantages[0, 0] - adv_0) < 1e-5

    def test_returns_equal_advantages_plus_values(self):
        buf = RolloutBuffer(3, 2, 4, 1)
        for i in range(3):
            buf.add(
                obs=np.random.randn(2, 4).astype(np.float32),
                actions=np.random.randn(2, 1).astype(np.float32),
                log_probs=np.random.randn(2).astype(np.float32),
                rewards=np.random.randn(2).astype(np.float32),
                terminateds=np.zeros(2, dtype=np.float32),
                values=np.random.randn(2).astype(np.float32),
            )
        buf.compute_advantages(
            np.random.randn(2).astype(np.float32),
            np.zeros(2, dtype=np.float32),
            gamma=0.99, gae_lambda=0.95,
        )
        np.testing.assert_allclose(buf.returns, buf.advantages + buf.values, atol=1e-5)

    def test_multi_env(self):
        """GAE should compute independently per environment."""
        buf = RolloutBuffer(2, 2, 1, 1)
        buf.add(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2),
                np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        buf.add(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2),
                np.array([3.0, 4.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        buf.compute_advantages(np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                               gamma=0.99, gae_lambda=0.95)
        # Env 0 and env 1 should have different advantages due to different rewards
        assert buf.advantages[0, 0] != buf.advantages[0, 1]


class TestMinibatches:
    def _fill_buffer(self, steps=4, envs=2, obs_dim=3, act_dim=1):
        buf = RolloutBuffer(steps, envs, obs_dim, act_dim)
        for i in range(steps):
            buf.add(
                obs=np.random.randn(envs, obs_dim).astype(np.float32),
                actions=np.random.randn(envs, act_dim).astype(np.float32),
                log_probs=np.random.randn(envs).astype(np.float32),
                rewards=np.random.randn(envs).astype(np.float32),
                terminateds=np.zeros(envs, dtype=np.float32),
                values=np.random.randn(envs).astype(np.float32),
            )
        buf.compute_advantages(
            np.zeros(envs, dtype=np.float32),
            np.zeros(envs, dtype=np.float32),
            gamma=0.99, gae_lambda=0.95,
        )
        return buf

    def test_covers_all_data(self):
        buf = self._fill_buffer(steps=8, envs=2)
        total = 8 * 2
        all_indices = set()
        for batch in buf.get_minibatches(num_minibatches=4):
            assert batch.obs.shape[0] == total // 4
            # Can't directly check indices, but verify total count
        batches = list(buf.get_minibatches(num_minibatches=4))
        total_samples = sum(b.obs.shape[0] for b in batches)
        assert total_samples == total

    def test_batch_shapes(self):
        buf = self._fill_buffer(steps=8, envs=2)
        for batch in buf.get_minibatches(num_minibatches=4):
            assert batch.obs.shape == (4, 3)
            assert batch.actions.shape == (4, 1)
            assert batch.old_log_probs.shape == (4,)
            assert batch.advantages.shape == (4,)
            assert batch.returns.shape == (4,)

    def test_advantages_normalized(self):
        buf = self._fill_buffer(steps=16, envs=4)
        for batch in buf.get_minibatches(num_minibatches=8):
            adv = batch.advantages.numpy()
            assert abs(adv.mean()) < 0.5  # roughly centered
