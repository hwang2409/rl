import torch

from network import ActorCritic, DiscreteActorCritic


class TestContinuousShapes:
    def test_forward_shapes(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(4, 17)
        dist, value = ac(obs)
        assert dist.sample().shape == (4, 6)
        assert value.shape == (4, 1)

    def test_get_action_and_value(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(4, 17)
        action, log_prob, entropy, value = ac.get_action_and_value(obs)
        assert action.shape == (4, 6)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_evaluate_actions(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(4, 17)
        actions = torch.randn(4, 6)
        log_prob, entropy, value = ac.evaluate_actions(obs, actions)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_single_obs(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(1, 17)
        action, log_prob, entropy, value = ac.get_action_and_value(obs)
        assert action.shape == (1, 6)
        assert log_prob.shape == (1,)


class TestDiscreteShapes:
    def test_forward_shapes(self):
        ac = DiscreteActorCritic(4, 2)
        obs = torch.randn(8, 4)
        dist, value = ac(obs)
        assert dist.sample().shape == (8,)
        assert value.shape == (8, 1)

    def test_get_action_and_value(self):
        ac = DiscreteActorCritic(4, 2)
        obs = torch.randn(8, 4)
        action, log_prob, entropy, value = ac.get_action_and_value(obs)
        assert action.shape == (8,)
        assert log_prob.shape == (8,)
        assert value.shape == (8,)


class TestDistribution:
    def test_entropy_positive(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(4, 17)
        _, _, entropy, _ = ac.get_action_and_value(obs)
        assert (entropy > 0).all()

    def test_small_initial_actions(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(100, 17)
        dist, _ = ac(obs)
        assert dist.mean.abs().mean() < 0.5


class TestGradients:
    def test_gradients_flow(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(4, 17)
        action, log_prob, entropy, value = ac.get_action_and_value(obs)
        loss = -log_prob.mean() + value.mean()
        loss.backward()
        for name, param in ac.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_value_unbounded(self):
        ac = ActorCritic(17, 6)
        obs = torch.randn(100, 17) * 10
        _, value = ac(obs)
        # Value should not be clamped to [-1, 1] (unlike AlphaZero)
        assert value.abs().max() > 0.01
