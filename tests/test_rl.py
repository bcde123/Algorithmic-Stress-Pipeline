"""
test_rl.py — Tests for the RL module: StressEnv, PPOPolicy, and PPO training.

Run with:
    pytest tests/test_rl.py -v
"""

import numpy as np
import pytest
import torch

try:
    import gymnasium  # noqa: F401
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _GYM_AVAILABLE,
    reason="gymnasium not installed — install with: pip install gymnasium"
)

from src.rl.stress_env import StressEnv
from src.rl.policy     import PPOPolicy
from src.rl.train_rl   import train_ppo, RolloutBuffer, compute_gae


N_FEATURES = 7
N_ACTIONS  = 5


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return StressEnv(n_features=N_FEATURES, n_actions=N_ACTIONS, max_steps=50, seed=0)


@pytest.fixture
def policy():
    return PPOPolicy(obs_dim=N_FEATURES, n_actions=N_ACTIONS, hidden_dims=[32, 16])


# ──────────────────────────────────────────────────────────────────────────────
# 1. StressEnv
# ──────────────────────────────────────────────────────────────────────────────

class TestStressEnv:

    def test_spaces(self, env):
        assert env.observation_space.shape == (N_FEATURES,)
        assert env.action_space.n          == N_ACTIONS

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (N_FEATURES,), f"Reset obs shape: {obs.shape}"
        assert isinstance(info, dict)

    def test_reset_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_returns_correct_shapes(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (N_FEATURES,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated,  bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self, env):
        """Episode must terminate within max_steps."""
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done  = terminated or truncated
            steps += 1
        assert done, "Episode never terminated"

    def test_high_load_raises_eda(self, env):
        """Persistent maximum-load actions should drive EDA upward."""
        obs, _ = env.reset()
        initial_eda = float(obs[0])
        for _ in range(30):
            obs, _, _, _, _ = env.step(N_ACTIONS - 1)  # max load
        final_eda = float(obs[0])
        assert final_eda > initial_eda, "Max load did not raise EDA"

    def test_rest_lowers_eda(self, env):
        """Rest actions (0) after stress should allow EDA recovery."""
        obs, _ = env.reset(seed=1)
        # Drive stress up first
        for _ in range(15):
            obs, _, _, _, _ = env.step(N_ACTIONS - 1)
        stressed_eda = float(obs[0])
        # Now rest
        for _ in range(15):
            obs, _, _, _, _ = env.step(0)
        recovered_eda = float(obs[0])
        assert recovered_eda < stressed_eda, "Rest did not reduce EDA"

    def test_data_stream_initialisation(self):
        """Env should use real data stream for initial obs when provided."""
        data_stream = np.random.default_rng(42).standard_normal((100, N_FEATURES)).astype(np.float32)
        env = StressEnv(data_stream=data_stream, seed=7)
        obs, _ = env.reset()
        assert obs.shape == (N_FEATURES,)

    def test_render_does_not_crash(self, env):
        env.reset()
        env.step(1)
        env.render()  # Should print without raising


# ──────────────────────────────────────────────────────────────────────────────
# 2. PPOPolicy
# ──────────────────────────────────────────────────────────────────────────────

class TestPPOPolicy:

    def _obs_batch(self, b=8):
        return torch.randn(b, N_FEATURES)

    def test_forward_returns_distribution_and_value(self, policy):
        from torch.distributions import Categorical
        obs  = self._obs_batch()
        dist, value = policy(obs)
        assert isinstance(dist, Categorical)
        assert value.shape == (8, 1)

    def test_get_action_shape(self, policy):
        obs = self._obs_batch(4)
        action, lp, val = policy.get_action(obs)
        assert action.shape == (4,), f"Action shape: {action.shape}"
        assert lp.shape    == (4,), f"Log-prob shape: {lp.shape}"
        assert val.shape   == (4, 1)

    def test_action_in_valid_range(self, policy):
        obs = self._obs_batch(32)
        for _ in range(10):
            action, _, _ = policy.get_action(obs)
            assert (action >= 0).all() and (action < N_ACTIONS).all()

    def test_deterministic_action(self, policy):
        """Deterministic mode should give reproducible argmax actions."""
        obs = self._obs_batch(4)
        a1, _, _ = policy.get_action(obs, deterministic=True)
        a2, _, _ = policy.get_action(obs, deterministic=True)
        assert torch.equal(a1, a2)

    def test_evaluate_actions_shapes(self, policy):
        obs     = self._obs_batch(16)
        actions = torch.randint(0, N_ACTIONS, (16,))
        lp, val, ent = policy.evaluate_actions(obs, actions)
        assert lp.shape  == (16,)
        assert val.shape == (16, 1)
        assert ent.shape == (16,)

    def test_log_probs_non_positive(self, policy):
        """Log probabilities of a valid dist must be ≤ 0."""
        obs     = self._obs_batch(16)
        actions = torch.randint(0, N_ACTIONS, (16,))
        lp, _, _ = policy.evaluate_actions(obs, actions)
        assert (lp <= 0).all(), "Log probs must be ≤ 0"

    def test_backward_pass(self, policy):
        obs     = self._obs_batch()
        actions = torch.randint(0, N_ACTIONS, (8,))
        lp, val, ent = policy.evaluate_actions(obs, actions)
        loss = -lp.mean() + 0.5 * val.mean() - 0.01 * ent.mean()
        loss.backward()
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. compute_gae
# ──────────────────────────────────────────────────────────────────────────────

class TestGAE:

    def test_returns_shapes(self):
        T       = 64
        rewards = torch.ones(T)
        values  = torch.zeros(T)
        dones   = torch.zeros(T)
        adv, ret = compute_gae(rewards, values, dones, last_value=0.0)
        assert adv.shape == (T,)
        assert ret.shape == (T,)

    def test_no_nan(self):
        T       = 32
        rewards = torch.randn(T)
        values  = torch.randn(T)
        dones   = (torch.rand(T) > 0.9).float()
        adv, ret = compute_gae(rewards, values, dones, last_value=0.5)
        assert not torch.isnan(adv).any()
        assert not torch.isnan(ret).any()


# ──────────────────────────────────────────────────────────────────────────────
# 4. train_ppo (smoke test — minimal steps)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainPPO:

    def test_training_runs(self, env, policy):
        """PPO training should complete without error and return history."""
        history = train_ppo(
            env, policy,
            total_steps=128,
            rollout_steps=64,
            n_epochs=2,
            verbose=False,
        )
        assert 'episode_rewards' in history
        assert 'policy_loss'     in history
        assert 'value_loss'      in history
        assert len(history['policy_loss']) > 0

    def test_history_losses_finite(self, env, policy):
        history = train_ppo(
            env, policy,
            total_steps=128,
            rollout_steps=64,
            n_epochs=1,
            verbose=False,
        )
        for loss_val in history['policy_loss']:
            assert np.isfinite(loss_val), f"Non-finite policy loss: {loss_val}"
        for loss_val in history['value_loss']:
            assert np.isfinite(loss_val),  f"Non-finite value loss: {loss_val}"

    def test_policy_weights_update(self, env):
        """Weights should change after at least one PPO update."""
        policy = PPOPolicy(obs_dim=N_FEATURES, n_actions=N_ACTIONS, hidden_dims=[32])
        before = [p.clone().detach() for p in policy.parameters()]
        train_ppo(env, policy, total_steps=128, rollout_steps=64, n_epochs=2, verbose=False)
        after  = [p.clone().detach() for p in policy.parameters()]
        any_changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert any_changed, "Policy parameters did not change after training"
