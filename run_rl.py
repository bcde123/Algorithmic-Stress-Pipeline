from src.rl.stress_env import StressEnv
from src.rl.policy import PPOPolicy
from src.rl.train_rl import train_ppo

def main():
    print("=" * 60)
    print("  Initializing Deep Reinforcement Learning Intervention")
    print("=" * 60)
    
    # Initialize the custom gymnasium environment and PPO actor-critic network
    env = StressEnv()
    policy = PPOPolicy(obs_dim=7, n_actions=5)
    
    # Train the reinforcement learning agent
    result = train_ppo(env, policy, total_steps=50000)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
