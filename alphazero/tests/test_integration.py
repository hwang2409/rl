import numpy as np
import torch

from config import Config
from game import Connect4
from model import AlphaZeroNet
from self_play import play_game, TrainingExample
from train import ReplayBuffer, train_step
from evaluate import RandomAgent, OneStepLookahead, MinimaxAgent, play_evaluation_game, MCTSAgent


class TestSelfPlayData:
    def test_produces_valid_examples(self):
        config = Config(num_simulations=10)
        model = AlphaZeroNet()
        examples = play_game(model, config)

        assert len(examples) > 0
        for ex in examples:
            assert ex.state.shape == (3, 6, 7)
            assert ex.policy.shape == (7,)
            assert abs(ex.policy.sum() - 1.0) < 1e-4
            assert -1.0 <= ex.value <= 1.0

    def test_augmentation_doubles_data(self):
        config = Config(num_simulations=10)
        model = AlphaZeroNet()
        examples = play_game(model, config)
        # augmentation doubles, so count should be even
        assert len(examples) % 2 == 0


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(1000)
        examples = [
            TrainingExample(
                state=np.random.randn(3, 6, 7).astype(np.float32),
                policy=np.array([1/7]*7, dtype=np.float32),
                value=0.0,
            )
            for _ in range(100)
        ]
        buf.add(examples)
        assert len(buf) == 100

        states, policies, values = buf.sample(16)
        assert states.shape == (16, 3, 6, 7)
        assert policies.shape == (16, 7)
        assert values.shape == (16, 1)

    def test_max_size(self):
        buf = ReplayBuffer(50)
        examples = [
            TrainingExample(
                state=np.zeros((3, 6, 7), dtype=np.float32),
                policy=np.array([1/7]*7, dtype=np.float32),
                value=0.0,
            )
            for _ in range(100)
        ]
        buf.add(examples)
        assert len(buf) == 50


class TestTrainStep:
    def test_loss_is_finite(self):
        model = AlphaZeroNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        states = torch.randn(8, 3, 6, 7)
        policies = torch.softmax(torch.randn(8, 7), dim=1)
        values = torch.randn(8, 1).clamp(-1, 1)

        total, p_loss, v_loss = train_step(model, optimizer, states, policies, values)
        assert np.isfinite(total)
        assert np.isfinite(p_loss)
        assert np.isfinite(v_loss)


class TestBaselineAgents:
    def test_random_agent_plays_valid(self):
        agent = RandomAgent()
        g = Connect4()
        for _ in range(20):
            if g.is_terminal()[0]:
                break
            action = agent.select_action(g)
            assert g.get_valid_moves()[action]
            g = g.make_move(action)

    def test_lookahead_takes_winning_move(self):
        agent = OneStepLookahead()
        g = Connect4()
        # Build a position where player 1 can win at col 3
        g = g.make_move(0).make_move(6).make_move(1).make_move(6).make_move(2)
        # Player -1 moves
        g = g.make_move(5)
        # Player 1 to move, can win at col 3
        action = agent.select_action(g)
        assert action == 3

    def test_lookahead_blocks_opponent(self):
        agent = OneStepLookahead()
        g = Connect4()
        # O has 3 in a row: cols 0,1,2
        g = g.make_move(6).make_move(0).make_move(6).make_move(1).make_move(5).make_move(2)
        # Player 1 to move, should block col 3
        action = agent.select_action(g)
        assert action == 3

    def test_minimax_plays_valid(self):
        agent = MinimaxAgent(depth=2)
        g = Connect4()
        for _ in range(10):
            if g.is_terminal()[0]:
                break
            action = agent.select_action(g)
            assert g.get_valid_moves()[action]
            g = g.make_move(action)

    def test_full_game_completes(self):
        result = play_evaluation_game(RandomAgent(), RandomAgent())
        assert result in (-1, 0, 1)


class TestMiniTraining:
    def test_two_iterations(self):
        """Smoke test: 2 iterations, 2 games, 10 sims. Verify loss decreases or stays reasonable."""
        config = Config(
            num_iterations=2,
            games_per_iteration=2,
            num_simulations=10,
            batch_size=8,
            train_epochs=2,
            min_replay_size=1,
            eval_games=2,
        )
        model = AlphaZeroNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        buf = ReplayBuffer(1000)

        losses = []
        for i in range(config.num_iterations):
            examples = play_game(model, config)
            buf.add(examples)
            if len(buf) >= config.min_replay_size:
                batch_size = min(config.batch_size, len(buf))
                states, policies, values = buf.sample(batch_size)
                total, _, _ = train_step(model, optimizer, states, policies, values)
                losses.append(total)

        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)
