"""
train_rl.py — PPO Training Loop for StressEnv

Implements Proximal Policy Optimisation (PPO-clip) in pure PyTorch — no
stable-baselines3 dependency.  Designed to run with the StressEnv gymnasium
environment to learn adaptive work-design policies that minimise sustained
physiological stress accumulation.

Algorithm summary (PPO-clip)
──────────────────────────────
1. Collect T steps of experience using current policy π_θ_old.
2. Compute advantages via Generalised Advantage Estimation (GAE).
3. Run K epochs of mini-batch gradient updates, clipping the probability ratio
   r_t = π_θ / π_θ_old within [1−ε, 1+ε].
4. Loss = -L_clip + c1 * L_value - c2 * L_entropy

Reference: Schulman et al. 2017 (https://arxiv.org/abs/1707.06347)

Usage
─────
    from src.rl.stress_env import StressEnv
    from src.rl.policy     import PPOPolicy
    from src.rl.train_rl   import train_ppo

    env    = StressEnv()
    policy = PPOPolicy(obs_dim=7, n_actions=5)
    result = train_ppo(env, policy, total_steps=50_000)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .stress_env import StressEnv
from .policy     import PPOPolicy


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores a single on-policy rollout (T steps) for PPO updates."""

    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.rewards:   List[float]      = []
        self.log_probs: List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def add(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def to_tensors(self, device: str = 'cpu'):
        return (
            torch.tensor(np.array(self.obs),      dtype=torch.float32).to(device),
            torch.tensor(self.actions,             dtype=torch.long   ).to(device),
            torch.tensor(self.rewards,             dtype=torch.float32).to(device),
            torch.tensor(self.log_probs,           dtype=torch.float32).to(device),
            torch.tensor(self.values,              dtype=torch.float32).to(device),
            torch.tensor(self.dones,               dtype=torch.float32).to(device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# GAE advantage computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:    torch.Tensor,
    values:     torch.Tensor,
    dones:      torch.Tensor,
    last_value: float,
    gamma:      float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation.

    Returns:
        advantages : (T,) advantage estimates.
        returns    : (T,) discounted returns (advantages + values).
    """
    advantages = torch.zeros_like(rewards)
    gae        = 0.0

    for t in reversed(range(len(rewards))):
        next_val  = last_value if t == len(rewards) - 1 else float(values[t + 1])
        delta     = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae       = float(delta) + gamma * gae_lambda * (1 - float(dones[t])) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO update step
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    policy:      PPOPolicy,
    optimizer:   optim.Optimizer,
    obs:         torch.Tensor,
    actions:     torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:  torch.Tensor,
    returns:     torch.Tensor,
    clip_eps:    float = 0.2,
    vf_coef:     float = 0.5,
    ent_coef:    float = 0.01,
    n_epochs:    int   = 4,
    mini_batch:  int   = 64,
) -> Dict[str, float]:
    """Runs K epochs of PPO-clip mini-batch updates."""
    n         = len(obs)
    metrics   = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    n_updates = 0

    for _ in range(n_epochs):
        indices = torch.randperm(n)
        for start in range(0, n, mini_batch):
            idx   = indices[start: start + mini_batch]
            mb_obs    = obs[idx]
            mb_act    = actions[idx]
            mb_oldlp  = old_log_probs[idx]
            mb_adv    = advantages[idx]
            mb_ret    = returns[idx]

            # Normalise advantages within mini-batch
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            new_lp, new_val, entropy = policy.evaluate_actions(mb_obs, mb_act)

            # PPO-clip objective
            ratio         = torch.exp(new_lp - mb_oldlp)
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss   = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_val.squeeze(), mb_ret)

            # Entropy bonus
            ent_loss = entropy.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss']  += value_loss.item()
            metrics['entropy']     += ent_loss.item()
            n_updates              += 1

    if n_updates > 0:
        for k in metrics:
            metrics[k] /= n_updates
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    env:           StressEnv,
    policy:        PPOPolicy,
    total_steps:   int   = 50_000,
    rollout_steps: int   = 512,
    lr:            float = 3e-4,
    gamma:         float = 0.99,
    gae_lambda:    float = 0.95,
    clip_eps:      float = 0.2,
    vf_coef:       float = 0.5,
    ent_coef:      float = 0.01,
    n_epochs:      int   = 4,
    mini_batch:    int   = 64,
    device:        str   = 'cpu',
    log_interval:  int   = 10,
    verbose:       bool  = True,
) -> Dict:
    """
    PPO training loop for the StressEnv environment.

    Args:
        env           : StressEnv instance (gymnasium-compatible).
        policy        : PPOPolicy actor-critic network.
        total_steps   : total environment interaction steps (not episodes).
        rollout_steps : steps per rollout collection phase (T).
        lr            : Adam learning rate.
        gamma         : discount factor.
        gae_lambda    : GAE lambda parameter.
        clip_eps      : PPO clip epsilon.
        vf_coef       : value function loss coefficient.
        ent_coef      : entropy bonus coefficient.
        n_epochs      : PPO update epochs per rollout.
        mini_batch    : mini-batch size for PPO updates.
        device        : 'cpu' or 'cuda'.
        log_interval  : print every N rollout updates.
        verbose       : whether to print progress.

    Returns:
        history dict with keys: 'episode_rewards', 'policy_loss', 'value_loss', 'entropy'.
    """
    policy    = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer    = RolloutBuffer()

    history: Dict[str, List] = {
        'episode_rewards': [],
        'policy_loss':     [],
        'value_loss':      [],
        'entropy':         [],
    }

    obs, _            = env.reset()
    ep_reward         = 0.0
    total_step_count  = 0
    n_updates         = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PPO Training — StressEnv")
        print(f"  Total steps: {total_steps:,}  |  Rollout: {rollout_steps}")
        print(f"{'='*60}")

    while total_step_count < total_steps:
        # ── Collect rollout ──────────────────────────────────────────────
        buffer.clear()
        for _ in range(rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_t, lp_t, val_t = policy.get_action(obs_t)

            action   = int(action_t.item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done     = terminated or truncated

            buffer.add(
                obs      = obs,
                action   = action,
                reward   = reward,
                log_prob = float(lp_t.item()),
                value    = float(val_t.item()),
                done     = float(done),
            )

            ep_reward        += reward
            total_step_count += 1

            if done:
                history['episode_rewards'].append(ep_reward)
                ep_reward = 0.0
                obs, _    = env.reset()
            else:
                obs = next_obs

            if total_step_count >= total_steps:
                break

        # ── Compute advantages ────────────────────────────────────────────
        obs_t, act_t, rew_t, lp_t, val_t, done_t = buffer.to_tensors(device)

        # Bootstrap value from last state
        with torch.no_grad():
            last_obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, last_val = policy.forward(last_obs_t)
            last_value  = float(last_val.item())

        advantages, returns = compute_gae(rew_t, val_t, done_t, last_value, gamma, gae_lambda)

        # ── PPO update ────────────────────────────────────────────────────
        metrics = ppo_update(
            policy, optimizer, obs_t, act_t, lp_t, advantages, returns,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
            n_epochs=n_epochs, mini_batch=mini_batch,
        )
        history['policy_loss'].append(metrics['policy_loss'])
        history['value_loss'].append(metrics['value_loss'])
        history['entropy'].append(metrics['entropy'])
        n_updates += 1

        if verbose and n_updates % log_interval == 0:
            mean_ep_rew = (
                np.mean(history['episode_rewards'][-20:])
                if history['episode_rewards'] else float('nan')
            )
            print(
                f"  Step {total_step_count:>7,} | Update {n_updates:>4} | "
                f"MeanEpRew={mean_ep_rew:+.3f} | "
                f"π_loss={metrics['policy_loss']:+.4f} | "
                f"V_loss={metrics['value_loss']:.4f} | "
                f"H={metrics['entropy']:.4f}"
            )

    if verbose:
        print(f"\n  Training complete — {n_updates} PPO updates over {total_step_count:,} steps.")

    return history
