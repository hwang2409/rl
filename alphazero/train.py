import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from config import Config
from evaluate import evaluate_against_baselines
from model import AlphaZeroNet
from self_play import TrainingExample, generate_self_play_data
from utils import AverageMeter, format_time, get_device, set_seed


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, examples: list[TrainingExample]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = torch.tensor(np.array([e.state for e in batch]), dtype=torch.float32)
        policies = torch.tensor(np.array([e.policy for e in batch]), dtype=torch.float32)
        values = torch.tensor(np.array([e.value for e in batch]), dtype=torch.float32).unsqueeze(-1)
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)


def train_step(model: AlphaZeroNet, optimizer: Adam,
               states: torch.Tensor, target_policies: torch.Tensor,
               target_values: torch.Tensor) -> tuple[float, float, float]:
    """Single training step. Returns (total_loss, policy_loss, value_loss)."""
    model.train()
    policy_logits, values = model(states)

    # Policy loss: cross-entropy with soft targets
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.mean(torch.sum(target_policies * log_probs, dim=1))

    # Value loss: MSE
    value_loss = F.mse_loss(values, target_values)

    total_loss = policy_loss + value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


def train(config: Config | None = None):
    if config is None:
        config = Config()

    device = get_device(config.device)
    print(f"Using device: {device}")
    set_seed(42)

    model = AlphaZeroNet(
        rows=config.rows, cols=config.cols,
        num_res_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = Adam(model.parameters(), lr=config.learning_rate,
                     weight_decay=config.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones,
                            gamma=config.lr_gamma)

    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    os.makedirs("checkpoints", exist_ok=True)
    total_start = time.time()

    for iteration in range(1, config.num_iterations + 1):
        iter_start = time.time()

        # Self-play (on CPU — single-sample inference is faster without CUDA overhead)
        print(f"\n--- Iteration {iteration}/{config.num_iterations} ---")
        print(f"Generating {config.games_per_iteration} self-play games...")
        sp_start = time.time()
        model.cpu()
        examples = generate_self_play_data(model, config, device="cpu")
        model.to(device)
        sp_time = time.time() - sp_start
        replay_buffer.add(examples)
        print(f"  Self-play: {format_time(sp_time)} | "
              f"Games: {config.games_per_iteration} | "
              f"Examples: {len(examples)} | "
              f"Buffer: {len(replay_buffer):,}")

        # Training
        if len(replay_buffer) < config.min_replay_size:
            print(f"  Buffer too small ({len(replay_buffer)} < {config.min_replay_size}), skipping training")
            continue

        loss_meter = AverageMeter()
        policy_meter = AverageMeter()
        value_meter = AverageMeter()

        for epoch in range(config.train_epochs):
            num_batches = max(1, len(replay_buffer) // config.batch_size)
            for _ in range(num_batches):
                batch_size = min(config.batch_size, len(replay_buffer))
                states, policies, values = replay_buffer.sample(batch_size)
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                total_loss, p_loss, v_loss = train_step(
                    model, optimizer, states, policies, values
                )
                loss_meter.update(total_loss)
                policy_meter.update(p_loss)
                value_meter.update(v_loss)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"  Train loss: {loss_meter.avg:.3f} "
              f"(policy: {policy_meter.avg:.3f}, value: {value_meter.avg:.3f}) | "
              f"LR: {lr:.6f}")

        # Evaluate every 5 iterations (and on the last one)
        if iteration % 5 == 0 or iteration == config.num_iterations:
            print("  Evaluating...")
            model.cpu()
            evaluate_against_baselines(model, config, device="cpu")
            model.to(device)

        # Save checkpoint
        checkpoint = {
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
        }
        path = f"checkpoints/iter_{iteration:03d}.pt"
        torch.save(checkpoint, path)

        iter_time = time.time() - iter_start
        total_time = time.time() - total_start
        print(f"  Iteration: {format_time(iter_time)} | Total: {format_time(total_time)}")

    print(f"\nTraining complete! Total time: {format_time(time.time() - total_start)}")
    return model


if __name__ == "__main__":
    train()
