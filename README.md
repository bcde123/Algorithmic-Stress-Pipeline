# Algorithmic Stress Monitoring Pipeline

A comprehensive machine learning and research-design framework leveraging physiological data (Empatica E4, etc.) to analyze stress, burnout, and algorithmic pressure in supply chain and knowledge work.

The contribution is organized around organizational behavior and labor-economics theory: algorithmic work demands create physiological strain, recovery resources moderate that strain, and sustained imbalance can translate into productivity, retention, and worker-welfare outcomes.

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

## Supported Datasets & Their Purposes

This framework leverages multiple real-world benchmark datasets, each serving a specific role in validating different aspects of algorithmic stress monitoring and intervention contrasts:

- **WESAD (Wearable Stress and Affect Detection):** Primary dataset for empirical validation of the pipeline. It provides high-fidelity, synchronized physiological signals (EDA, BVP, TEMP, ACC) and labeled baseline/stress/recovery-like conditions used for prediction metrics and intervention-style contrasts.
- **Induced Stress (PhysioNet Wearable Dataset):** Used for distinguishing cognitive stress from physical exertion. By comparing stress sessions with aerobic/anaerobic exercise sessions, the pipeline trains exertion-filtering components and tests whether high-demand exposure changes physiological outcomes after adjusting for physical movement.
- **MMASH (Multilevel Monitoring of Activity and Sleep in Healthy people):** Provides 24-hour continuous monitoring data (beat-to-beat RR intervals and Actigraphy) essential for **Pillar 1: Long-Term Dynamics**. It supports recovery-resource and circadian-imbalance constructs.
- **SWELL Knowledge Work:** Tailored for evaluating stress in knowledge work environments. It contains RRI streams and precomputed HRV features collected during no-stress, interruption, and time-pressure work conditions, directly supporting managerial intervention validation.

---

## Running the Models

This project comprises several analytical models that formulate an integrated pipeline. The framework now separates prediction from research design and relies on six core pillars:
- **Pillar 1: Long-Term Dynamics** (parsimonious AttentionLSTM / CircadianAttentionLSTM)
- **Pillar 2: Short-Term Reactions** (parsimonious StressTCN)
- **Pillar 3: Latent State Detection** (compact StressAutoencoder)
- **Pillar 4: Psychological Signatures** (K-Means Clustering on Attention Profiles)
- **Pillar 5: Attrition / Burnout Risk** (DeepSurv and Cox PH baseline)
- **Pillar 6: Research Design** (OB/labor-economics theory, causal estimands, intervention validation, robustness, external validation, and managerial implications)

### 1. End-To-End Main Pipeline (Pillars 1-6)

You can run the full integrated pipeline that trains the measurement models and writes the research-design summary. If no data directories are provided, a synthetic fallback dataset will be auto-generated to validate model functionality.

**Using synthetic mock data:**
```bash
python -m src.main
```

**Using real datasets:**
You can pass directories containing the physiological datasets (WESAD, Induced Stress, MMASH, SWELL) via environment variables. The integrated data loader will harmonize signals and apply exertion filtering automatically.
```bash
WESAD_DIR=data/raw/wesad/WESAD INDUCED_DIR="data/raw/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1" MMASH_DIR=data/raw/mmash SWELL_DIR=data/raw/swell python -m src.main
```

> Output visualizations, threshold reports, and `research_design_summary.txt` will be saved into the `outputs/` directory.

### 2. Research Design Outputs

The end-to-end pipeline writes a dedicated `outputs/research_design_summary.txt` file that includes:
- **Theoretical framework:** Job Demands-Resources, Demand-Control, Effort-Recovery, and labor-economics constructs mapped to physiological variables.
- **Causal estimands:** Adjusted high-stress exposure effects on outcomes such as HRV and HR, with bootstrap confidence intervals and explicit assumptions.
- **Intervention validation:** High-demand versus recovery/control contrasts from source intervention labels where available.
- **Robustness checks:** Threshold-sensitivity analysis and leave-one-dataset-out external validation.
- **Policy implications:** Managerial guidance on pacing, breaks, task rotation, interruption reduction, and deployment safeguards.

### 3. Deep Reinforcement Learning Intervention Model

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

### 4. Individual Model Architectures

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
pytest tests/ -v
# OR to run simply:
pytest
```
