# Algorithmic Stress Monitoring Pipeline

A comprehensive machine learning framework leveraging physiological data (Empatica E4, etc.) to analyze stress, burnout, and algorithmic pressure in supply chain and knowledge work.

## Setup & Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Models

This project comprises several analytical models that formulate an integrated pipeline. The framework relies on five core pillars:
- **Pillar 1: Long-Term Dynamics** (AttentionLSTM / CircadianAttentionLSTM)
- **Pillar 2: Short-Term Reactions** (StressTCN)
- **Pillar 3: Latent State Detection** (StressAutoencoder)
- **Pillar 4: Psychological Signatures** (K-Means Clustering on Attention Profiles)
- **Pillar 5: Attrition / Burnout Risk** (DeepSurv)

### 1. End-To-End Main Pipeline (Pillars 1-5)

You can run the full integrated pipeline that trains and demonstrates all five pillars sequentially. If no data directories are provided, a synthetic fallback dataset will be auto-generated to validate model functionality.

**Using synthetic mock data:**
```bash
python -m src.main
```

**Using real datasets:**
You can pass directories containing the physiological datasets (WESAD, Induced Stress, MMASH, SWELL) via environment variables. The integrated data loader will harmonize signals and apply exertion filtering automatically.
```bash
WESAD_DIR=data/raw/wesad/WESAD INDUCED_DIR="data/raw/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1" MMASH_DIR=data/raw/mmash SWELL_DIR=data/raw/swell python -m src.main
```

> Output visualizations and threshold reports will be saved into the `outputs/` directory.

### 2. Deep Reinforcement Learning Intervention Model

Complementing the measurement pillars is a reinforcement learning module designed to adaptively alter work policies to reduce physiological stress accumulation. It uses custom Proximal Policy Optimization (PPO).

You can run this RL training pipeline by executing the following directly in Python:

```python
from src.rl.stress_env import StressEnv
from src.rl.policy import PPOPolicy
from src.rl.train_rl import train_ppo

# Initialize the custom gymnasium environment and Proximal Policy Optimization (PPO) actor-critic network
env = StressEnv()
policy = PPOPolicy(obs_dim=7, n_actions=5)

# Train the reinforcement learning agent
result = train_ppo(env, policy, total_steps=50000)
```

```bash
python run_rl.py
```

### 3. Individual Model Architectures

If you wish to evaluate or embed specific model components independently, they can be found within the `src.models` module and trained using the loops established in the `src.training` directory:
- **LSTM / Sequential Context:** `src.models.lstm` & `src.training.train_lstm`
- **Temporal ConvNet:** `src.models.tcn` & `src.training.train_tcn`
- **Autoencoder Anomalies:** `src.models.autoencoder` & `src.training.train_autoencoder`
- **Survival Analysis:** `src.models.deepsurv` & `src.training.train_survival`
- **Deep ANN:** `src.models.ann`

---

## Running Assessments & Tests

This repository is instrumented with rigorous integration validations. To run the automated tests against your local environment setup:

```bash
pytest testing/ -v
# OR to run simply:
pytest
```
