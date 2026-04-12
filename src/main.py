"""
main.py — End-to-End Algorithmic Stress Monitoring Pipeline

Demonstrates all four analytical pillars against the integrated dataset:

  Pillar 0: Data Ingestion & Harmonization  — IntegratedLoader
  Pillar 1: Long-Term Dynamics              — CircadianAttentionLSTM
  Pillar 2: Short-Term Reactions            — StressTCN (multi-head)
  Pillar 3: Latent State Detection          — StressAutoencoder + RegimeDetector
  Pillar 4: Psychological Signatures        — SignatureAnalyzer (K-Means on attention)

The pipeline works in two modes:
  1. REAL DATA — when dataset paths are configured via DataConfig.
  2. SYNTHETIC FALLBACK — when no paths are available, a representative mock
     DataFrame is generated so the modelling logic can still be validated.

Usage:
    python -m src.main
    # or with real data:
    WESAD_DIR=/path/to/WESAD INDUCED_DIR=/path/to/induced python -m src.main
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Ensure the src/ directory is on the path for relative imports
_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Data layer ────────────────────────────────────────────────────────────────
from data.integrated_loader  import IntegratedLoader, DataConfig, FEATURE_COLS
from data.transforms         import apply_butter_bandpass, normalize_zscore
from data.dataset            import PhysiologicalTimeSeriesDataset
from data.survival_dataset   import SyntheticSurvivalDataset

# ── Models ────────────────────────────────────────────────────────────────────
from models.autoencoder import StressAutoencoder
from models.lstm        import AttentionLSTM, CircadianAttentionLSTM
from models.tcn         import StressTCN
from models.deepsurv    import DeepSurv
from models.ann         import StressMLP

# ── Training ──────────────────────────────────────────────────────────────────
from training.train_autoencoder import train_autoencoder, calculate_reconstruction_error
from training.train_lstm        import train_attention_lstm
from training.train_tcn         import train_tcn
from training.train_survival    import train_survival_model

# ── Analysis & Visualization ──────────────────────────────────────────────────
from analysis.visualizer import (
    plot_stress_distribution,
    plot_regime_timeline,
    plot_signature_heatmap,
    plot_pressure_nodes,
    plot_circadian_curve,
)
from analysis.report_generator import ThresholdReportGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_EPOCHS   = 3     # quick validation; increase for real training
SEQ_LEN    = 60    # seconds (at 1 Hz)
STRIDE     = 10
BATCH_SIZE = 1024
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Build synthetic fallback DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_df(n: int = 2000) -> pd.DataFrame:
    """Creates a synthetic but representative physiological stream."""
    rng = np.random.default_rng(42)
    t   = np.arange(n)

    # Simulate a stress ramp then recovery
    stress_profile = np.clip(np.sin(2 * np.pi * t / (n * 0.6)) * 0.5 + 0.5, 0, 1)

    df = pd.DataFrame({
        'EDA':   rng.normal(0.5, 0.1, n) + stress_profile * 0.3,
        'HR':    rng.normal(70, 5, n)    + stress_profile * 15,
        'TEMP':  rng.normal(36.5, 0.2, n),
        'ACC_x': rng.normal(0, 1, n),
        'ACC_y': rng.normal(0, 1, n),
        'ACC_z': rng.normal(0, 1, n),
        'HRV':   rng.normal(0.05, 0.01, n) - stress_profile * 0.02,
        'stress_index': stress_profile,
        'dataset':    'Synthetic',
        'subject_id': 'mock_01',
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ALGORITHMIC STRESS MONITORING — FULL PIPELINE")
    print(f"  Device: {DEVICE.upper()}")
    print("=" * 70)

    # ── PILLAR 0: Data Ingestion & Harmonization ──────────────────────────────
    print("\n[PILLAR 0] Integrated Data Ingestion & Harmonization")

    config = DataConfig(
        wesad_dir          = os.environ.get('WESAD_DIR'),
        induced_stress_dir = os.environ.get('INDUCED_DIR'),
        mmash_dir          = os.environ.get('MMASH_DIR'),
        swell_dir          = os.environ.get('SWELL_DIR'),
        apply_exertion_filter = True,
    )

    has_real_data = any([
        config.wesad_dir, config.induced_stress_dir,
        config.mmash_dir, config.swell_dir,
    ])

    if has_real_data:
        loader = IntegratedLoader(config)
        try:
            df = loader.combine()
            print(f"  Loaded {len(df):,} rows from real datasets: "
                  f"{df['dataset'].value_counts().to_dict()}")
        except RuntimeError as e:
            print(f"  Warning: {e} — falling back to synthetic data.")
            df = _make_synthetic_df()
    else:
        print("  No real dataset paths set — using synthetic fallback.")
        df = _make_synthetic_df()

    if len(df) > 10000:
        print(f"  Dataset is extremely large ({len(df):,} rows). Taking 10% subset for training...")
        # Take the first contiguous 10% of each dataset to preserve time-series integrity
        df = df.groupby('dataset', group_keys=False).apply(lambda x: x.iloc[:int(len(x) * 0.1)]).reset_index(drop=True)

    print(f"  DataFrame shape: {df.shape}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Stress index range: [{df['stress_index'].min():.3f}, {df['stress_index'].max():.3f}]")

    # Visualize stress distribution
    if 'dataset' in df.columns:
        plot_stress_distribution(
            df, save_path=str(OUTPUT_DIR / "stress_distribution.png")
        )
        print("  [Saved] stress_distribution.png")

    # ── Biological transforms ─────────────────────────────────────────────────
    print("\n[PRE-PROC] Applying biological transforms …")
    # Bandpass filter is only valid when fs > 2 * highcut (Nyquist theorem).
    # After harmonization all signals are at 1 Hz, so the 0.5 Hz highcut is
    # at the Nyquist limit and invalid. We skip the filter and rely on the
    # exertion filter + Z-score normalization instead.
    # (When working with native 4 Hz EDA, call apply_butter_bandpass(…, fs=4).)
    df[FEATURE_COLS] = normalize_zscore(df[FEATURE_COLS])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    print("  Z-score normalization applied (bandpass skipped at 1 Hz).")


    # ── Sliding-window datasets ───────────────────────────────────────────────
    print("\n[DATA] Building sliding-window PyTorch datasets …")

    # Unsupervised (Autoencoder, Pillar 3)
    ae_dataset = PhysiologicalTimeSeriesDataset(
        df[FEATURE_COLS], sequence_length=SEQ_LEN, stride=STRIDE, target_col=None
    )
    ae_train_sz = int(0.8 * len(ae_dataset))
    ae_train, ae_val = torch.utils.data.random_split(
        ae_dataset, [ae_train_sz, len(ae_dataset) - ae_train_sz]
    )
    ae_train_loader = DataLoader(ae_train, batch_size=BATCH_SIZE, shuffle=True)
    ae_val_loader   = DataLoader(ae_val,   batch_size=BATCH_SIZE, shuffle=False)

    # Supervised (LSTM & TCN, Pillars 1 & 2)
    sup_df = df[FEATURE_COLS + ['stress_index']].copy()
    lstm_dataset = PhysiologicalTimeSeriesDataset(
        sup_df, sequence_length=SEQ_LEN, stride=STRIDE, target_col='stress_index'
    )
    lstm_train_sz = int(0.8 * len(lstm_dataset))
    lstm_train, lstm_val = torch.utils.data.random_split(
        lstm_dataset, [lstm_train_sz, len(lstm_dataset) - lstm_train_sz]
    )
    lstm_train_loader = DataLoader(lstm_train, batch_size=BATCH_SIZE, shuffle=True)
    lstm_val_loader   = DataLoader(lstm_val,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"  AE windows : {len(ae_dataset):,}  |  LSTM windows: {len(lstm_dataset):,}")

    # ── PILLAR 3: Deep Autoencoder ────────────────────────────────────────────
    print("\n[PILLAR 3] Training Deep Autoencoder (Latent Regime Detector) …")
    autoencoder = StressAutoencoder(input_dim=len(FEATURE_COLS), latent_dim=16)
    ae_config   = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2}
    trained_ae, ae_history = train_autoencoder(
        autoencoder, ae_train_loader, ae_val_loader, ae_config
    )
    errors = calculate_reconstruction_error(trained_ae, ae_val_loader)
    threshold = float(np.percentile(errors, 95))
    flags     = (np.array(errors) > threshold).astype(int)

    print(f"  Reconstruction errors computed. Threshold (95th pct): {threshold:.5f}")
    print(f"  Latent Stress Regime windows detected: {flags.sum()} / {len(flags)}")

    plot_regime_timeline(
        np.array(errors), threshold, flags,
        save_path=str(OUTPUT_DIR / "regime_timeline.png")
    )
    print("  [Saved] regime_timeline.png")

    # ── PILLAR 1a: AttentionLSTM (short windows) ──────────────────────────────
    print("\n[PILLAR 1a] Training Attention-LSTM (standard windows) …")
    lstm_model  = AttentionLSTM(input_dim=len(FEATURE_COLS), hidden_dim=32, num_layers=2, output_dim=1)
    lstm_config = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2}
    trained_lstm, lstm_history = train_attention_lstm(
        lstm_model, lstm_train_loader, lstm_val_loader, lstm_config
    )

    # Collect attention weights for Pillar 4
    trained_lstm.eval()
    attn_profiles = []
    with torch.no_grad():
        for batch_x, _ in lstm_val_loader:
            _, attn = trained_lstm(batch_x)
            weighted = torch.sum(attn * batch_x, dim=1).cpu().numpy()
            attn_profiles.append(weighted)
    attn_profiles = np.concatenate(attn_profiles, axis=0)
    print(f"  Collected {len(attn_profiles)} attention-weighted profiles.")

    # ── PILLAR 1b: CircadianAttentionLSTM — supervised training ───────────────
    print("\n[PILLAR 1b] Training CircadianAttentionLSTM (circadian dynamics) …")
    # Note: a fully meaningful circadian encoding requires 24-hour (86 400-step)
    # sequences; here we train on 60-second windows as a pipeline verification.
    circ_model = CircadianAttentionLSTM(
        input_dim=len(FEATURE_COLS), embed_dim=32, hidden_dim=32,
        num_layers=2, output_dim=1, max_seq_len=SEQ_LEN + 100
    )
    circ_config = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2}
    trained_circ, circ_history = train_attention_lstm(
        circ_model, lstm_train_loader, lstm_val_loader, circ_config
    )
    # Visualise a single forward pass for reporting
    trained_circ.eval()
    with torch.no_grad():
        demo_x          = torch.randn(1, SEQ_LEN, len(FEATURE_COLS))
        s_idx, imbal, attn_w = trained_circ(demo_x)
    print(f"  Demo fwd pass — Stress Index: {s_idx.item():.4f}  |  Imbalance: {imbal.item():.4f}")
    plot_circadian_curve(
        stress_index = s_idx.squeeze().expand(SEQ_LEN).cpu().numpy(),
        imbalance    = imbal.squeeze().expand(SEQ_LEN).cpu().numpy(),
        attn_weights = attn_w.squeeze().cpu().numpy(),
        title        = "CircadianAttentionLSTM — trained output (short-window proxy)",
        save_path    = str(OUTPUT_DIR / "circadian_curve.png"),
    )
    print("  [Saved] circadian_curve.png")

    # ── PILLAR 2: StressTCN (multi-head) ─────────────────────────────────────
    print("\n[PILLAR 2] Training StressTCN (Short-Term Reactions) …")
    tcn_model  = StressTCN(
        input_dim=len(FEATURE_COLS), num_channels=[32, 32], output_dim=1, num_fragments=5
    )
    tcn_config = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2,
                  'alpha': 0.7, 'beta': 0.3}
    trained_tcn, tcn_history = train_tcn(
        tcn_model, lstm_train_loader, lstm_val_loader, tcn_config, device=DEVICE
    )
    print(f"  TCN training complete. Best val loss: {min(tcn_history['val_loss']):.4f}")

    # ── PILLAR 4: Psychological Signatures ───────────────────────────────────
    print("\n[PILLAR 4] Clustering Psychological Signatures …")
    from sklearn.cluster import KMeans
    n_clusters = 3
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters   = kmeans.fit_predict(attn_profiles)
    centers    = kmeans.cluster_centers_

    PSYCH_STATES = [
        "Loss of Agency / Cognitive Overload",
        "Identity Instability / High Agitation",
        "Stable Algorithmic Pacing",
    ]

    plot_signature_heatmap(
        centers, FEATURE_COLS, PSYCH_STATES,
        save_path=str(OUTPUT_DIR / "signature_heatmap.png")
    )
    print("  [Saved] signature_heatmap.png")

    plot_pressure_nodes(
        attn_profiles, clusters, PSYCH_STATES,
        save_path=str(OUTPUT_DIR / "pressure_nodes.png")
    )
    print("  [Saved] pressure_nodes.png")

    print(f"\n  Cluster distribution: {dict(zip(*np.unique(clusters, return_counts=True)))}")

    # ── PILLAR 5: DeepSurv — Attrition / Burnout Risk ─────────────────────────
    print("\n[PILLAR 5] Training DeepSurv (Attrition & Burnout Risk) …")
    surv_dataset = SyntheticSurvivalDataset(
        df, feature_cols=FEATURE_COLS, window_size=SEQ_LEN,
        burnout_threshold=0.6, max_time=180.0
    )
    surv_train_sz = int(0.8 * len(surv_dataset))
    surv_train, surv_val = torch.utils.data.random_split(
        surv_dataset, [surv_train_sz, len(surv_dataset) - surv_train_sz]
    )
    surv_train_loader = DataLoader(surv_train, batch_size=BATCH_SIZE, shuffle=True)
    surv_val_loader   = DataLoader(surv_val,   batch_size=BATCH_SIZE, shuffle=False)

    deepsurv_model = DeepSurv(input_dim=len(FEATURE_COLS), hidden_layers=[64, 32])
    surv_config    = {'epochs': N_EPOCHS, 'learning_rate': 1e-4}
    trained_surv, surv_history = train_survival_model(
        deepsurv_model, surv_train_loader, surv_val_loader, surv_config
    )

    # Compute per-subject mean risk score from the trained model
    trained_surv.eval()
    all_risks = []
    with torch.no_grad():
        for covs, _, _ in surv_val_loader:
            risk = trained_surv(covs).squeeze().cpu().numpy()
            all_risks.extend(risk.tolist() if risk.ndim > 0 else [float(risk)])
    mean_risk = float(np.mean(all_risks)) if all_risks else 0.0
    print(f"  DeepSurv training complete. Mean val risk score: {mean_risk:.4f}")

    # ── Threshold Report ──────────────────────────────────────────────────────
    print("\n[REPORT] Generating threshold report …")
    reporter = ThresholdReportGenerator()
    report   = reporter.build_subject_report(
        subject_id   = df['subject_id'].iloc[0] if 'subject_id' in df.columns else 'unknown',
        ae_errors    = errors,
        ae_threshold = threshold,
        imbalance    = float(imbal.item()),
        risk_score   = mean_risk,
    )
    report.consecutive_breaches = ThresholdReportGenerator.compute_consecutive_breaches(
        np.array(flags)
    )
    report_text = reporter.format_report(report)
    print(report_text)
    reporter.save_report(report, path=str(OUTPUT_DIR / 'threshold_report.txt'))
    print("  [Saved] threshold_report.txt")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Outputs written to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
