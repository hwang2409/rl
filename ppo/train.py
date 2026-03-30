import os
import time

import gymnasium as gym
import numpy as np
import torch

from config import PPOConfig, cartpole_config, halfcheetah_config
from evaluate import evaluate, record_video
from network import ActorCritic, DiscreteActorCritic
from ppo import PPO
from rollout_buffer import RolloutBuffer
from utils import format_time, get_device, set_seed


def make_envs(config: PPOConfig) -> gym.vector.SyncVectorEnv:
    """Create vectorized environments.

    N environments step in parallel, giving N times more data per
    wall-clock second. Each auto-resets independently when an episode ends.
    """
    def make_env(seed: int):
        def _init():
            env = gym.make(config.env_name)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed)
            return env
        return _init

    return gym.vector.SyncVectorEnv(
        [make_env(config.seed + i) for i in range(config.num_envs)]
    )


def train(config: PPOConfig | None = None):
    if config is None:
        config = PPOConfig()

    device = get_device(config.device)
    print(f"Using device: {device}")
    print(f"Environment: {config.env_name}")
    set_seed(config.seed)

    envs = make_envs(config)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    obs_dim = obs_space.shape[0]

    discrete = isinstance(act_space, gym.spaces.Discrete)
    if discrete:
        act_dim = 1  # for buffer storage
        ac = DiscreteActorCritic(obs_dim, act_space.n, config.hidden_dim).to(device)
    else:
        act_dim = act_space.shape[0]
        ac = ActorCritic(obs_dim, act_dim, config.hidden_dim).to(device)

    param_count = sum(p.numel() for p in ac.parameters())
    print(f"Model parameters: {param_count:,}")

    ppo = PPO(ac, config)
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, obs_dim, act_dim)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training state
    obs, _ = envs.reset(seed=config.seed)
    steps_per_update = config.rollout_steps * config.num_envs
    num_updates = config.total_timesteps // steps_per_update
    global_step = 0
    episode_returns = []
    total_start = time.time()

    for update in range(1, num_updates + 1):
        update_start = time.time()

        # Anneal learning rate
        if config.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            ppo.optimizer.param_groups[0]["lr"] = config.learning_rate * frac

        # === Collect rollouts ===
        ac.eval()
        for step in range(config.rollout_steps):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                action, log_prob, _, value = ac.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            if discrete:
                env_action = action_np
                store_action = action_np.reshape(-1, 1)
            else:
                env_action = action_np
                store_action = action_np

            next_obs, reward, terminated, truncated, infos = envs.step(env_action)

            buffer.add(obs, store_action, log_prob.cpu().numpy(),
                       reward, terminated.astype(np.float32),
                       value.cpu().numpy())

            obs = next_obs
            global_step += config.num_envs

            # Track completed episodes (gymnasium 1.x format)
            if "_episode" in infos:
                for i, done_flag in enumerate(infos["_episode"]):
                    if done_flag:
                        episode_returns.append(float(infos["episode"]["r"][i]))

        # === Compute advantages ===
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            last_value = ac.critic(obs_t).squeeze(-1).cpu().numpy() if not discrete else \
                ac.critic(obs_t).squeeze(-1).cpu().numpy()
        last_terminated = terminated.astype(np.float32)
        buffer.compute_advantages(last_value, last_terminated,
                                  config.gamma, config.gae_lambda)

        # === PPO update ===
        ac.train()
        metrics = ppo.update(buffer, device=device)
        buffer.reset()

        # === Logging ===
        lr = ppo.optimizer.param_groups[0]["lr"]
        update_time = time.time() - update_start
        total_time = time.time() - total_start

        print(f"\n--- Update {update}/{num_updates} ({global_step:,} steps) ---")
        if episode_returns:
            recent = episode_returns[-20:]
            print(f"  Episodes: {len(episode_returns)} | "
                  f"Avg return: {np.mean(recent):.1f} | "
                  f"Last: {episode_returns[-1]:.1f}")
        print(f"  PPO: policy={metrics['policy_loss']:.4f} "
              f"value={metrics['value_loss']:.4f} "
              f"entropy={metrics['entropy']:.3f} "
              f"kl={metrics['approx_kl']:.4f} "
              f"clip={metrics['clip_fraction']:.3f}")
        print(f"  LR: {lr:.6f} | "
              f"Update: {format_time(update_time)} | "
              f"Total: {format_time(total_time)}")

        # === Evaluation ===
        if global_step % config.eval_interval < steps_per_update:
            print("  Evaluating...")
            eval_returns = evaluate(ac, config, device, discrete=discrete)
            print(f"  Eval: {np.mean(eval_returns):.1f} +/- {np.std(eval_returns):.1f}")

        # === Checkpoint ===
        if update % 5 == 0 or update == num_updates:
            checkpoint = {
                "update": update,
                "global_step": global_step,
                "model_state": ac.state_dict(),
                "optimizer_state": ppo.optimizer.state_dict(),
                "config": config,
            }
            path = os.path.join(config.checkpoint_dir, f"step_{global_step:07d}.pt")
            torch.save(checkpoint, path)

    envs.close()
    print(f"\nTraining complete! Total time: {format_time(time.time() - total_start)}")
    return ac


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cartpole":
        train(cartpole_config())
    elif len(sys.argv) > 1 and sys.argv[1] == "halfcheetah":
        train(halfcheetah_config())
    else:
        train()
