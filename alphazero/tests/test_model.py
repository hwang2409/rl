import numpy as np
import torch

from game import Connect4
from model import AlphaZeroNet


class TestOutputShapes:
    def test_forward_shapes(self):
        net = AlphaZeroNet()
        x = torch.randn(4, 3, 6, 7)
        policy, value = net(x)
        assert policy.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_single_input(self):
        net = AlphaZeroNet()
        x = torch.randn(1, 3, 6, 7)
        policy, value = net(x)
        assert policy.shape == (1, 7)
        assert value.shape == (1, 1)


class TestValueRange:
    def test_value_bounded(self):
        net = AlphaZeroNet()
        x = torch.randn(32, 3, 6, 7)
        _, value = net(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()


class TestPredict:
    def test_predict_shapes(self):
        net = AlphaZeroNet()
        g = Connect4()
        policy, value = net.predict(g)
        assert policy.shape == (7,)
        assert isinstance(value, float)

    def test_policy_sums_to_one(self):
        net = AlphaZeroNet()
        g = Connect4()
        policy, _ = net.predict(g)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_policy_masks_full_column(self):
        net = AlphaZeroNet()
        g = Connect4()
        for _ in range(6):
            g = g.make_move(0)
        policy, _ = net.predict(g)
        assert policy[0] == 0.0
        assert abs(policy[1:].sum() - 1.0) < 1e-5

    def test_deterministic_eval_mode(self):
        net = AlphaZeroNet()
        g = Connect4().make_move(3)
        p1, v1 = net.predict(g)
        p2, v2 = net.predict(g)
        np.testing.assert_array_equal(p1, p2)
        assert v1 == v2


class TestGradients:
    def test_gradients_flow(self):
        net = AlphaZeroNet()
        x = torch.randn(4, 3, 6, 7)
        policy, value = net(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None


class TestParameterCount:
    def test_reasonable_size(self):
        net = AlphaZeroNet()
        total = sum(p.numel() for p in net.parameters())
        assert 100_000 < total < 1_000_000
