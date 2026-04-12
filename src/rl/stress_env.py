"""
stress_env.py — StressEnv: Gymnasium Environment for Adaptive Work-Design RL

Models a worker exposed to algorithmic work assignment.  The agent controls
the task-intensity level (action) and the environment propagates the
physiological state in response, computed via a learned or heuristic
physiological dynamics model.

State space
───────────
A fixed-length window of the 7 canonical physiological features:
    [EDA, HR, TEMP, ACC_x, ACC_y, ACC_z, HRV]
Shape: (n_features,)  — mean-pooled over the current episode window.

Action space
─────────────
Discrete, 5 levels representing work-intensity assignments:
    0 = Rest / Recovery period
    1 = Light cognitive load
    2 = Moderate cognitive load
    3 = High cognitive load (algorithmically fragmented tasks)
    4 = Maximum load (continuous monitoring / high-pressure deadlines)

Reward function
────────────────
The reward is a recovery-rate proxy that incentivises the agent to
reduce physiological stress while maintaining an acceptable productivity level:

    r = α * (Δ EDA improvement) + β * (Δ HRV improvement) - γ * load_penalty

where:
    Δ EDA improvement = baseline_EDA - current_EDA   (high EDA = bad)
    Δ HRV improvement = current_HRV - baseline_HRV   (low HRV  = bad)
    load_penalty      = action / (n_actions - 1)      (higher action = harder)
    α = 0.4, β = 0.4, γ = 0.2

Episode termination
────────────────────
An episode ends when:
    • max_steps steps are taken (typical: 480 = 8-hour shift at 1 Hz per minute);
    • or EDA exceeds ``burnout_eda_threshold`` for ``burnout_steps`` consecutive
      steps (simulating clinical exhaustion).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required for StressEnv.  Install it with:\n"
        "    pip install gymnasium"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Physiological dynamics model (heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def _physio_step(
    state: np.ndarray,
    action: int,
    n_actions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Applies a heuristic physiological dynamics model.

    Higher actions increase EDA/HR ergodically and suppress HRV; lower actions
    allow recovery toward resting baseline.  Small Gaussian noise is added to
    simulate inter-individual variability.
    """
    new_state = state.copy()
    load      = action / max(n_actions - 1, 1)   # normalised [0, 1]

    # Indices: [EDA, HR, TEMP, ACC_x, ACC_y, ACC_z, HRV]
    EDA, HR, TEMP, ACC_X, ACC_Y, ACC_Z, HRV = range(7)

    # EDA: rises with load, slow recovery
    new_state[EDA]  = np.clip(state[EDA]  + 0.05 * load - 0.02 * (1 - load) + rng.normal(0, 0.01), -3, 3)
    # HR: proportional to load
    new_state[HR]   = np.clip(state[HR]   + 0.10 * load - 0.05 * (1 - load) + rng.normal(0, 0.02), -3, 3)
    # TEMP: slow drift, minimally affected by short-term load
    new_state[TEMP] = np.clip(state[TEMP] + 0.01 * load - 0.005             + rng.normal(0, 0.005), -3, 3)
    # ACC: proportional to load (physical task activity proxy)
    for ax in (ACC_X, ACC_Y, ACC_Z):
        new_state[ax] = np.clip(state[ax] + 0.05 * load * rng.standard_normal() + rng.normal(0, 0.01), -3, 3)
    # HRV: inversely proportional to load
    new_state[HRV]  = np.clip(state[HRV]  - 0.04 * load + 0.02 * (1 - load) + rng.normal(0, 0.01), -3, 3)

    return new_state.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# StressEnv
# ─────────────────────────────────────────────────────────────────────────────

class StressEnv(gym.Env):
    """
    Gymnasium environment simulating a worker under algorithmic work assignment.

    Args:
        n_features           : dimensionality of the physiological state (7).
        n_actions            : number of discrete task-intensity levels (5).
        max_steps            : maximum episode length (default 480 = 8-hour proxy).
        burnout_eda_threshold: EDA z-score above which burnout risk is flagged.
        burnout_steps        : consecutive high-EDA steps before forced termination.
        alpha                : weight for EDA improvement in reward.
        beta                 : weight for HRV improvement in reward.
        gamma                : weight for load penalty in reward.
        seed                 : RNG seed.
        data_stream          : optional (N, n_features) array of real physiological
                               data used to initialise episode states.  If None,
                               states are sampled from N(0, 1).
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        n_features:            int   = 7,
        n_actions:             int   = 5,
        max_steps:             int   = 480,
        burnout_eda_threshold: float = 2.5,
        burnout_steps:         int   = 10,
        alpha:                 float = 0.4,
        beta:                  float = 0.4,
        gamma:                 float = 0.2,
        seed:                  Optional[int] = None,
        data_stream:           Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_features            = n_features
        self.n_actions             = n_actions
        self.max_steps             = max_steps
        self.burnout_eda_threshold = burnout_eda_threshold
        self.burnout_steps         = burnout_steps
        self.alpha                 = alpha
        self.beta                  = beta
        self.gamma                 = gamma
        self.data_stream           = data_stream

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

        self._rng = np.random.default_rng(seed)
        self._state:            np.ndarray = np.zeros(n_features, dtype=np.float32)
        self._baseline:         np.ndarray = np.zeros(n_features, dtype=np.float32)
        self._step_count:       int        = 0
        self._burnout_counter:  int        = 0

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.data_stream is not None:
            # Sample a random starting row from the real data stream
            idx = self._rng.integers(0, len(self.data_stream))
            self._state = self.data_stream[idx].astype(np.float32).copy()
        else:
            self._state = self._rng.standard_normal(self.n_features).astype(np.float32)

        self._baseline        = self._state.copy()
        self._step_count      = 0
        self._burnout_counter = 0
        return self._state.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_state = self._state.copy()
        self._state = _physio_step(self._state, action, self.n_actions, self._rng)
        self._step_count += 1

        # ── Reward ───────────────────────────────────────────────────────────
        eda_improvement = float(self._baseline[0] - self._state[0])
        hrv_improvement = float(self._state[6]    - self._baseline[6])
        load_penalty    = action / max(self.n_actions - 1, 1)
        reward = (
            self.alpha * eda_improvement
            + self.beta * hrv_improvement
            - self.gamma * load_penalty
        )

        # ── Termination conditions ─────────────────────────────────────────
        # Burnout: sustained high EDA
        if self._state[0] > self.burnout_eda_threshold:
            self._burnout_counter += 1
        else:
            self._burnout_counter = 0

        terminated = bool(self._burnout_counter >= self.burnout_steps)
        truncated  = bool(self._step_count >= self.max_steps)

        info = {
            'step':            self._step_count,
            'eda':             float(self._state[0]),
            'hrv':             float(self._state[6]),
            'burnout_counter': self._burnout_counter,
            'reward':          reward,
        }
        return self._state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> None:
        print(
            f"Step {self._step_count:4d} | "
            f"EDA={self._state[0]:+.3f}  HR={self._state[1]:+.3f}  "
            f"HRV={self._state[6]:+.3f}  "
            f"Burnout_counter={self._burnout_counter}"
        )
