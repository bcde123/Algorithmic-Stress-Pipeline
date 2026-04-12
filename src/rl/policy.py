"""
policy.py — PPOPolicy: Actor-Critic Network for StressEnv

Implements a shared-backbone actor-critic network used by the PPO training
loop.  Both the actor (action distribution) and the critic (value estimate)
share a common MLP feature extractor to promote sample-efficient learning.

Architecture
────────────
    Input (obs) → SharedBackbone → ┬→ Actor head  → action logits (n_actions,)
                                   └→ Critic head → value scalar  (1,)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOPolicy(nn.Module):
    """
    Shared actor-critic policy for the StressEnv PPO agent.

    Args:
        obs_dim       : dimension of the observation (physiological state).
        n_actions     : number of discrete actions (task-intensity levels).
        hidden_dims   : widths of the shared backbone layers.
        actor_hidden  : width of the actor-specific head layer.
        critic_hidden : width of the critic-specific head layer.
    """

    def __init__(
        self,
        obs_dim:        int,
        n_actions:       int,
        hidden_dims:    List[int] = None,
        actor_hidden:   int       = 64,
        critic_hidden:  int       = 64,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # ── Shared backbone ────────────────────────────────────────────────
        layers: List[nn.Module] = []
        current = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(current, h), nn.Tanh()]
            current  = h
        self.backbone = nn.Sequential(*layers)

        # ── Actor head ─────────────────────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(current, actor_hidden),
            nn.Tanh(),
            nn.Linear(actor_hidden, n_actions),   # raw logits → Categorical
        )

        # ── Critic head ────────────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(current, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, 1),           # value estimate
        )

        # Orthogonal initialisation improves PPO stability
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Returns the action distribution and value estimate for a given observation.

        Args:
            obs : (batch, obs_dim) float32 tensor.
        Returns:
            dist  : Categorical distribution over actions.
            value : (batch, 1) value estimate.
        """
        features = self.backbone(obs)
        logits   = self.actor(features)
        value    = self.critic(features)
        dist     = Categorical(logits=logits)
        return dist, value

    # ------------------------------------------------------------------
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) an action for rollout collection.

        Returns:
            action     : scalar action tensor.
            log_prob   : log probability of the selected action.
            value      : critic value estimate.
        """
        dist, value = self.forward(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value

    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities, values, and entropy for a batch of (obs, action)
        pairs collected during rollout.  Used in the PPO update step.

        Returns:
            log_probs : (batch,) log probabilities of the taken actions.
            values    : (batch, 1) value estimates.
            entropy   : (batch,) action entropy (for the entropy bonus).
        """
        dist, values = self.forward(obs)
        log_probs    = dist.log_prob(actions)
        entropy      = dist.entropy()
        return log_probs, values, entropy
