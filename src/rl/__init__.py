"""RL sub-package for adaptive work-design intervention modeling.

Import sub-modules directly to avoid hard dependency on gymnasium at
import time for codebases that don't use the RL component:

    from src.rl.stress_env import StressEnv
    from src.rl.policy     import PPOPolicy
    from src.rl.train_rl   import train_ppo
"""

