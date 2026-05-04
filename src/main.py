"""
main.py — End-to-End Algorithmic Stress Monitoring Pipeline

Demonstrates the analytical and research-design pillars against the integrated dataset:

  Pillar 0: Data Ingestion & Harmonization  — IntegratedLoader
  Pillar 1: Long-Term Dynamics              — CircadianAttentionLSTM
  Pillar 2: Short-Term Reactions            — StressTCN (multi-head)
  Pillar 3: Latent State Detection          — StressAutoencoder + RegimeDetector
  Pillar 4: Psychological Signatures        — SignatureAnalyzer (K-Means on attention)
  Pillar 5: Attrition / Burnout Risk        — DeepSurv
  Pillar 6: Research Design                 — Theory, causality, robustness, policy

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
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
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
from analysis.research_design import (
    build_research_design_summary,
    format_research_design_summary,
)

# ── Evaluation metrics ────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    silhouette_score, davies_bouldin_score,
    normalized_mutual_info_score,
)
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _select_device() -> str:
    requested = os.environ.get("ASMP_DEVICE")
    if requested:
        requested = requested.lower()
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ASMP_DEVICE=cuda was requested, but torch.cuda.is_available() is false.")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


N_EPOCHS   = _env_int("ASMP_EPOCHS", 3)     # quick validation; increase for real training
SEQ_LEN    = _env_int("ASMP_SEQ_LEN", 60)    # seconds (at 1 Hz)
STRIDE     = _env_int("ASMP_STRIDE", 10)
BATCH_SIZE = _env_int("ASMP_BATCH_SIZE", 1024)
DEVICE     = _select_device()
NUM_WORKERS = _env_int("ASMP_NUM_WORKERS", 0)
PIN_MEMORY = DEVICE == "cuda"
EVAL_MAX_WINDOWS = _env_int("ASMP_EVAL_MAX_WINDOWS", 5000)
MAX_ROWS = _env_int("ASMP_MAX_ROWS", 0)
SAMPLE_FRACTION = _env_float("ASMP_SAMPLE_FRACTION", 1.0 if DEVICE == "cuda" else 0.1)
AUTO_DISCOVER_DATA = _env_flag("ASMP_AUTO_DISCOVER_DATA", False)
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
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
    intervention = np.where(
        stress_profile > 0.65,
        'high_workload',
        np.where(stress_profile < 0.35, 'recovery', 'standard_work')
    )

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
        'intervention': intervention,
    })
    return df


def _default_data_path(*parts: str):
    path = ROOT_DIR.joinpath(*parts)
    return str(path) if path.exists() else None


def _build_loader_kwargs(name: str) -> dict:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {name.lower().replace("asmp_", ""): values}


def _make_loader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def _maybe_limit_windows(loader: DataLoader, max_windows: int = EVAL_MAX_WINDOWS):
    seen = 0
    for batch in loader:
        batch_size = len(batch[0])
        if max_windows > 0 and seen >= max_windows:
            break
        seen += batch_size
        yield batch


def _history_min(history: dict, key: str) -> float:
    values = history.get(key, [])
    return float(np.nanmin(values)) if values else float("nan")


def _serializable_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ALGORITHMIC STRESS MONITORING — FULL PIPELINE")
    print(f"  Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── PILLAR 0: Data Ingestion & Harmonization ──────────────────────────────
    print("\n[PILLAR 0] Integrated Data Ingestion & Harmonization")

    config = DataConfig(
        wesad_dir          = os.environ.get('WESAD_DIR') or (_default_data_path("data", "raw", "wesad") if AUTO_DISCOVER_DATA else None),
        induced_stress_dir = os.environ.get('INDUCED_DIR') or (_default_data_path(
            "data", "raw", "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1",
            "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1",
        ) if AUTO_DISCOVER_DATA else None),
        mmash_dir          = os.environ.get('MMASH_DIR') or (_default_data_path("data", "raw", "mmash") if AUTO_DISCOVER_DATA else None),
        swell_dir          = os.environ.get('SWELL_DIR') or (_default_data_path("data", "raw", "swell") if AUTO_DISCOVER_DATA else None),
        **_build_loader_kwargs("ASMP_WESAD_SUBJECTS"),
        **_build_loader_kwargs("ASMP_INDUCED_STRESS_SUBJECTS"),
        **_build_loader_kwargs("ASMP_MMASH_USERS"),
        **_build_loader_kwargs("ASMP_SWELL_SUBJECTS"),
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

    original_rows = len(df)
    if has_real_data and SAMPLE_FRACTION < 1.0:
        print(f"  Dataset has {len(df):,} rows. Taking {SAMPLE_FRACTION:.0%} contiguous subset for this run...")
        df = df.groupby('dataset', group_keys=False).apply(lambda x: x.iloc[:max(SEQ_LEN + 1, int(len(x) * SAMPLE_FRACTION))]).reset_index(drop=True)
    if MAX_ROWS > 0 and len(df) > MAX_ROWS:
        print(f"  Limiting dataset from {len(df):,} to ASMP_MAX_ROWS={MAX_ROWS:,} rows.")
        df = df.groupby('dataset', group_keys=False).apply(lambda x: x.iloc[:max(SEQ_LEN + 1, int(MAX_ROWS * len(x) / len(df)))]).reset_index(drop=True)

    if 'intervention' not in df.columns:
        df['intervention'] = 'unlabeled'

    print(f"  DataFrame shape: {df.shape}")
    print(f"  Source rows before sampling: {original_rows:,}")
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
    ae_train_loader = _make_loader(ae_train, shuffle=True)
    ae_val_loader   = _make_loader(ae_val,   shuffle=False)

    # Supervised (LSTM & TCN, Pillars 1 & 2)
    sup_df = df[FEATURE_COLS + ['stress_index']].copy()
    lstm_dataset = PhysiologicalTimeSeriesDataset(
        sup_df, sequence_length=SEQ_LEN, stride=STRIDE, target_col='stress_index'
    )
    lstm_train_sz = int(0.8 * len(lstm_dataset))
    lstm_train, lstm_val = torch.utils.data.random_split(
        lstm_dataset, [lstm_train_sz, len(lstm_dataset) - lstm_train_sz]
    )
    lstm_train_loader = _make_loader(lstm_train, shuffle=True)
    lstm_val_loader   = _make_loader(lstm_val,   shuffle=False)

    print(f"  AE windows : {len(ae_dataset):,}  |  LSTM windows: {len(lstm_dataset):,}")

    # ── PILLAR 3: Deep Autoencoder ────────────────────────────────────────────
    print("\n[PILLAR 3] Training Parsimonious Autoencoder (Latent Regime Detector) …")
    autoencoder = StressAutoencoder(input_dim=len(FEATURE_COLS), latent_dim=8)
    ae_config   = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2}
    trained_ae, ae_history = train_autoencoder(
        autoencoder, ae_train_loader, ae_val_loader, ae_config, device=DEVICE
    )
    errors = calculate_reconstruction_error(trained_ae, ae_val_loader, device=DEVICE)
    threshold = float(np.percentile(errors, 95))
    flags     = (np.array(errors) > threshold).astype(int)

    print(f"  Reconstruction errors computed. Threshold (95th pct): {threshold:.5f}")
    print(f"  Latent Stress Regime windows detected: {flags.sum()} / {len(flags)}")

    plot_regime_timeline(
        np.array(errors), threshold, flags,
        save_path=str(OUTPUT_DIR / "regime_timeline.png")
    )
    print("  [Saved] regime_timeline.png")

    # ── Pillar 3 Evaluation: Precision/Recall vs IsolationForest baseline ────
    print("\n[EVAL P3] Autoencoder anomaly detection vs IsolationForest baseline …")
    # Reconstruct ground-truth for val windows from the supervised loader
    ae_true_stress = []
    for _, batch_y in _maybe_limit_windows(lstm_val_loader, max_windows=len(flags)):
        ae_true_stress.extend(batch_y.numpy().flatten().tolist())
    ae_true_arr = (np.array(ae_true_stress[:len(flags)]) > 0.5).astype(int)
    eval_flags = flags[:len(ae_true_arr)]

    ae_prec = ae_rec = if_prec = if_rec = float('nan')
    try:
        if len(np.unique(ae_true_arr)) > 1:
            ae_prec = precision_score(ae_true_arr, eval_flags, zero_division=0)
            ae_rec  = recall_score(ae_true_arr, eval_flags, zero_division=0)

            # IsolationForest baseline on flattened val windows
            ae_flat = []
            for batch_x, _ in _maybe_limit_windows(ae_val_loader):
                ae_flat.append(batch_x.numpy().reshape(len(batch_x), -1))
            ae_flat = np.concatenate(ae_flat, axis=0)
            iforest = IsolationForest(contamination=0.05, random_state=42)
            if_labels = (iforest.fit_predict(ae_flat) == -1).astype(int)
            eval_len = min(len(ae_true_arr), len(if_labels))
            if_prec = precision_score(ae_true_arr[:eval_len], if_labels[:eval_len], zero_division=0)
            if_rec  = recall_score(ae_true_arr[:eval_len], if_labels[:eval_len], zero_division=0)
    except Exception as e:
        print(f"  (Evaluation skipped: {e})")

    print(f"  Autoencoder    — Precision: {ae_prec:.4f}  |  Recall: {ae_rec:.4f}")
    print(f"  IsolationForest— Precision: {if_prec:.4f}  |  Recall: {if_rec:.4f}")

    # ── PILLAR 1a: AttentionLSTM (short windows) ──────────────────────────────
    print("\n[PILLAR 1a] Training Parsimonious Attention-LSTM (standard windows) …")
    lstm_model  = AttentionLSTM(input_dim=len(FEATURE_COLS), hidden_dim=16, num_layers=1, output_dim=1)
    lstm_config = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2}
    trained_lstm, lstm_history = train_attention_lstm(
        lstm_model, lstm_train_loader, lstm_val_loader, lstm_config, device=DEVICE
    )

    # Collect attention weights for Pillar 4 + predictions for evaluation
    trained_lstm.eval()
    attn_profiles, lstm_preds, lstm_trues = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in _maybe_limit_windows(lstm_val_loader):
            batch_x = batch_x.to(DEVICE)
            preds, attn = trained_lstm(batch_x)
            weighted = torch.sum(attn * batch_x, dim=1).cpu().numpy()
            attn_profiles.append(weighted)
            lstm_preds.extend(preds.squeeze(-1).cpu().numpy().tolist())
            lstm_trues.extend(batch_y.numpy().flatten().tolist())
    attn_profiles = np.concatenate(attn_profiles, axis=0)
    print(f"  Collected {len(attn_profiles)} attention-weighted profiles.")

    # ── Pillar 1a Evaluation: AUROC, F1 vs SVM baseline ─────────────────────
    print("\n[EVAL P1a] AttentionLSTM vs SVM baseline …")
    lstm_preds_arr = np.array(lstm_preds)
    lstm_trues_arr = np.array(lstm_trues)
    lstm_binary    = (lstm_preds_arr > 0.5).astype(int)
    true_binary    = (lstm_trues_arr > 0.5).astype(int)

    lstm_auroc = lstm_f1 = float('nan')
    svm_auroc  = svm_f1  = float('nan')
    try:
        if len(np.unique(true_binary)) > 1:
            lstm_auroc = roc_auc_score(true_binary, lstm_preds_arr)
            lstm_f1    = f1_score(true_binary, lstm_binary, zero_division=0)

            # SVM baseline: train on flattened val windows
            val_X_flat = np.array([b.numpy().flatten() for b, _ in _maybe_limit_windows(lstm_val_loader)
                                   for b in b])[:len(true_binary)]
            svm_pipe = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0))
            svm_pipe.fit(val_X_flat, lstm_trues_arr)
            svm_pred = svm_pipe.predict(val_X_flat)
            svm_binary = (svm_pred > 0.5).astype(int)
            if len(np.unique(true_binary)) > 1:
                svm_auroc = roc_auc_score(true_binary, np.clip(svm_pred, 0, 1))
                svm_f1    = f1_score(true_binary, svm_binary, zero_division=0)
    except Exception as e:
        print(f"  (Evaluation skipped: {e})")

    print(f"  AttentionLSTM — AUROC: {lstm_auroc:.4f}  |  F1: {lstm_f1:.4f}")
    print(f"  SVM baseline  — AUROC: {svm_auroc:.4f}  |  F1: {svm_f1:.4f}")

    # ── PILLAR 1b: CircadianAttentionLSTM — supervised training ───────────────
    print("\n[PILLAR 1b] Training CircadianAttentionLSTM (circadian dynamics) …")
    # Note: a fully meaningful circadian encoding requires 24-hour (86 400-step)
    # sequences; here we train on 60-second windows as a pipeline verification.
    circ_model = CircadianAttentionLSTM(
        input_dim=len(FEATURE_COLS), embed_dim=16, hidden_dim=16,
        num_layers=1, output_dim=1, max_seq_len=SEQ_LEN + 100
    )
    circ_config = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2}
    trained_circ, circ_history = train_attention_lstm(
        circ_model, lstm_train_loader, lstm_val_loader, circ_config, device=DEVICE
    )
    # Evaluate CircadianLSTM (previously had zero evaluation — only a random demo input)
    trained_circ.eval()
    circ_preds, circ_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in _maybe_limit_windows(lstm_val_loader):
            batch_x = batch_x.to(DEVICE)
            s_idx_b, imbal_b, attn_w_b = trained_circ(batch_x)
            circ_preds.extend(s_idx_b.squeeze(-1).cpu().numpy().tolist())
            circ_trues.extend(batch_y.numpy().flatten().tolist())
    # Use last batch for visualisation
    s_idx, imbal, attn_w = s_idx_b, imbal_b, attn_w_b
    print(f"  Demo fwd pass — Stress Index: {s_idx[0].item():.4f}  |  Imbalance: {imbal[0].item():.4f}")
    plot_circadian_curve(
        stress_index = np.resize(np.array(circ_preds), SEQ_LEN),
        imbalance    = np.resize(imbal.squeeze().cpu().numpy(), SEQ_LEN),
        attn_weights = np.resize(attn_w[0].squeeze().cpu().numpy(), SEQ_LEN),
        title        = "CircadianAttentionLSTM — trained output (short-window proxy)",
        save_path    = str(OUTPUT_DIR / "circadian_curve.png"),
    )
    print("  [Saved] circadian_curve.png")

    circ_preds_arr = np.array(circ_preds)
    circ_trues_arr = np.array(circ_trues)
    circ_binary    = (circ_preds_arr > 0.5).astype(int)
    circ_true_bin  = (circ_trues_arr > 0.5).astype(int)
    circ_auroc = circ_f1 = float('nan')
    try:
        if len(np.unique(circ_true_bin)) > 1:
            circ_auroc = roc_auc_score(circ_true_bin, circ_preds_arr)
            circ_f1    = f1_score(circ_true_bin, circ_binary, zero_division=0)
    except Exception:
        pass
    print(f"\n[EVAL P1b] CircadianLSTM — AUROC: {circ_auroc:.4f}  |  F1: {circ_f1:.4f}")

    # ── PILLAR 2: StressTCN (multi-head) ─────────────────────────────────────
    print("\n[PILLAR 2] Training StressTCN (Short-Term Reactions) …")
    tcn_model  = StressTCN(
        input_dim=len(FEATURE_COLS), num_channels=[16], output_dim=1, num_fragments=3
    )
    tcn_config = {'epochs': N_EPOCHS, 'learning_rate': 1e-3, 'patience': 2,
                  'alpha': 0.7, 'beta': 0.3}
    trained_tcn, tcn_history = train_tcn(
        tcn_model, lstm_train_loader, lstm_val_loader, tcn_config, device=DEVICE
    )
    print(f"  TCN training complete. Best val loss: {min(tcn_history['val_loss']):.4f}")

    # ── Pillar 2 Evaluation: AUROC, F1 ───────────────────────────────────────
    print("\n[EVAL P2] StressTCN evaluation …")
    trained_tcn.eval()
    tcn_preds, tcn_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in _maybe_limit_windows(lstm_val_loader):
            batch_x = batch_x.to(DEVICE)
            out = trained_tcn(batch_x)
            # TCN may return (pred, aux) tuple depending on training config
            pred = out[0] if isinstance(out, (tuple, list)) else out
            tcn_preds.extend(pred.squeeze(-1).cpu().numpy().tolist())
            tcn_trues.extend(batch_y.numpy().flatten().tolist())
    tcn_preds_arr = np.array(tcn_preds)
    tcn_trues_arr = np.array(tcn_trues)
    tcn_binary    = (tcn_preds_arr > 0.5).astype(int)
    tcn_true_bin  = (tcn_trues_arr > 0.5).astype(int)
    tcn_auroc = tcn_f1 = float('nan')
    try:
        if len(np.unique(tcn_true_bin)) > 1:
            tcn_auroc = roc_auc_score(tcn_true_bin, tcn_preds_arr)
            tcn_f1    = f1_score(tcn_true_bin, tcn_binary, zero_division=0)
    except Exception:
        pass
    print(f"  StressTCN — AUROC: {tcn_auroc:.4f}  |  F1: {tcn_f1:.4f}")

    # ── PILLAR 4: Psychological Signatures ───────────────────────────────────
    print("\n[PILLAR 4] Clustering Psychological Signatures …")
    from sklearn.cluster import KMeans
    n_clusters = 3
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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

    # Clean display of cluster distribution (avoid np.int32 repr)
    cluster_ids, cluster_counts = np.unique(clusters, return_counts=True)
    cluster_dist = {int(k): int(v) for k, v in zip(cluster_ids, cluster_counts)}
    print(f"\n  Cluster distribution: {cluster_dist}")

    # ── Pillar 4 Evaluation: Clustering quality metrics ───────────────────
    print("\n[EVAL P4] Clustering quality metrics …")
    if len(attn_profiles) > n_clusters:
        sil_score = silhouette_score(attn_profiles, clusters)
        db_score  = davies_bouldin_score(attn_profiles, clusters)

        # NMI: compare cluster labels vs binarised stress (if ground-truth exists)
        val_stress_labels = []
        for batch_x, batch_y in _maybe_limit_windows(lstm_val_loader):
            val_stress_labels.extend(batch_y.numpy().flatten().tolist())
        stress_binary = (np.array(val_stress_labels[:len(clusters)]) > 0.5).astype(int)
        nmi_score = normalized_mutual_info_score(stress_binary, clusters[:len(stress_binary)])

        print(f"  Silhouette score     : {sil_score:+.4f}  (higher → more separated clusters)")
        print(f"  Davies-Bouldin index : {db_score:.4f}   (lower → better defined clusters)")
        print(f"  NMI vs stress label  : {nmi_score:.4f}  (0=random, 1=perfect alignment)")
    else:
        sil_score = db_score = nmi_score = float('nan')
        print("  (Skipped — not enough samples for clustering metrics)")

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
    surv_train_loader = _make_loader(surv_train, shuffle=True)
    surv_val_loader   = _make_loader(surv_val,   shuffle=False)

    deepsurv_model = DeepSurv(input_dim=len(FEATURE_COLS), hidden_layers=[32, 16])
    surv_config    = {'epochs': N_EPOCHS, 'learning_rate': 1e-4}
    trained_surv, surv_history = train_survival_model(
        deepsurv_model, surv_train_loader, surv_val_loader, surv_config, device=DEVICE
    )

    # Compute per-subject mean risk score from the trained model
    trained_surv.eval()
    all_risks, all_durations, all_events = [], [], []
    with torch.no_grad():
        for covs, durations, events in _maybe_limit_windows(surv_val_loader):
            covs = covs.to(DEVICE)
            risk = trained_surv(covs).squeeze().cpu().numpy()
            all_risks.extend(risk.tolist() if risk.ndim > 0 else [float(risk)])
            all_durations.extend(durations.numpy().flatten().tolist())
            all_events.extend(events.numpy().flatten().tolist())
    mean_risk = float(np.mean(all_risks)) if all_risks else 0.0
    print(f"  DeepSurv training complete. Mean val risk score: {mean_risk:.4f}")

    # ── Pillar 5 Evaluation: C-index vs Cox PH baseline ──────────────────────
    print("\n[EVAL P5] DeepSurv C-index vs Cox PH baseline …")
    cindex_deepsurv = cindex_cox = float('nan')
    try:
        from lifelines import CoxPHFitter
        durations_arr = np.array(all_durations)
        events_arr    = np.array(all_events)
        risks_arr     = np.array(all_risks)

        if len(np.unique(events_arr)) > 1 and len(risks_arr) > 2:
            # DeepSurv C-index (higher risk → shorter time → concordant)
            cindex_deepsurv = concordance_index(durations_arr, -risks_arr, events_arr)

            # Cox PH baseline on mean features per window
            cox_features = []
            for covs, _, _ in _maybe_limit_windows(surv_val_loader):
                cox_features.append(covs.numpy())
            cox_features = np.concatenate(cox_features, axis=0)
            cox_features = cox_features[:len(durations_arr)]
            cox_df = pd.DataFrame(cox_features, columns=FEATURE_COLS)
            cox_df['duration'] = durations_arr[:len(cox_df)]
            cox_df['event']    = events_arr[:len(cox_df)]
            cox_df = cox_df.dropna()
            if len(cox_df) > 5:
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(cox_df, duration_col='duration', event_col='event')
                cox_risk = cph.predict_partial_hazard(cox_df).values
                cindex_cox = concordance_index(cox_df['duration'].values,
                                               -cox_risk, cox_df['event'].values)
    except Exception as e:
        print(f"  (C-index evaluation skipped: {e})")

    print(f"  DeepSurv   C-index: {cindex_deepsurv:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  Cox PH     C-index: {cindex_cox:.4f}")

    # ── Unified Metrics Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Pillar':<12} {'Model':<28} {'Metric':<12} {'Value':>8}  {'Baseline':>10}")
    print("  " + "-" * 68)
    print(f"  {'P1a LSTM':<12} {'AttentionLSTM':<28} {'AUROC':<12} {lstm_auroc:>8.4f}  {svm_auroc:>10.4f}")
    print(f"  {'P1a LSTM':<12} {'AttentionLSTM':<28} {'F1':<12} {lstm_f1:>8.4f}  {svm_f1:>10.4f}")
    print(f"  {'P1b Circ':<12} {'CircadianLSTM':<28} {'AUROC':<12} {circ_auroc:>8.4f}  {'—':>10}")
    print(f"  {'P1b Circ':<12} {'CircadianLSTM':<28} {'F1':<12} {circ_f1:>8.4f}  {'—':>10}")
    print(f"  {'P2 TCN':<12} {'StressTCN':<28} {'AUROC':<12} {tcn_auroc:>8.4f}  {'—':>10}")
    print(f"  {'P2 TCN':<12} {'StressTCN':<28} {'F1':<12} {tcn_f1:>8.4f}  {'—':>10}")
    print(f"  {'P3 AE':<12} {'StressAutoencoder':<28} {'Precision':<12} {ae_prec:>8.4f}  {if_prec:>10.4f}")
    print(f"  {'P3 AE':<12} {'StressAutoencoder':<28} {'Recall':<12} {ae_rec:>8.4f}  {if_rec:>10.4f}")
    print(f"  {'P4 KMeans':<12} {'K-Means (k=3)':<28} {'Silhouette':<12} {sil_score:>8.4f}  {'—':>10}")
    print(f"  {'P4 KMeans':<12} {'K-Means (k=3)':<28} {'DB-Index':<12} {db_score:>8.4f}  {'—':>10}")
    print(f"  {'P4 KMeans':<12} {'K-Means (k=3)':<28} {'NMI':<12} {nmi_score:>8.4f}  {'—':>10}")
    print(f"  {'P5 Surv':<12} {'DeepSurv':<28} {'C-index':<12} {cindex_deepsurv:>8.4f}  {cindex_cox:>10.4f}")
    print("  " + "-" * 68)
    print("  Baseline column: SVM (P1a) | IsolationForest (P3) | Cox PH (P5)")
    print("=" * 70)

    dataset_summary = (
        df.groupby("dataset")
        .agg(rows=("stress_index", "size"), mean_stress=("stress_index", "mean"), high_stress_rate=("stress_index", lambda x: float((x >= 0.6).mean())))
        .reset_index()
    )
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    evaluation_rows = [
        {"pillar": "P1a", "model": "AttentionLSTM", "metric": "AUROC", "value": lstm_auroc, "baseline_model": "SVM", "baseline_value": svm_auroc},
        {"pillar": "P1a", "model": "AttentionLSTM", "metric": "F1", "value": lstm_f1, "baseline_model": "SVM", "baseline_value": svm_f1},
        {"pillar": "P1b", "model": "CircadianLSTM", "metric": "AUROC", "value": circ_auroc, "baseline_model": None, "baseline_value": None},
        {"pillar": "P1b", "model": "CircadianLSTM", "metric": "F1", "value": circ_f1, "baseline_model": None, "baseline_value": None},
        {"pillar": "P2", "model": "StressTCN", "metric": "AUROC", "value": tcn_auroc, "baseline_model": None, "baseline_value": None},
        {"pillar": "P2", "model": "StressTCN", "metric": "F1", "value": tcn_f1, "baseline_model": None, "baseline_value": None},
        {"pillar": "P3", "model": "StressAutoencoder", "metric": "Precision", "value": ae_prec, "baseline_model": "IsolationForest", "baseline_value": if_prec},
        {"pillar": "P3", "model": "StressAutoencoder", "metric": "Recall", "value": ae_rec, "baseline_model": "IsolationForest", "baseline_value": if_rec},
        {"pillar": "P4", "model": "K-Means", "metric": "Silhouette", "value": sil_score, "baseline_model": None, "baseline_value": None},
        {"pillar": "P4", "model": "K-Means", "metric": "DB-Index", "value": db_score, "baseline_model": None, "baseline_value": None},
        {"pillar": "P4", "model": "K-Means", "metric": "NMI", "value": nmi_score, "baseline_model": None, "baseline_value": None},
        {"pillar": "P5", "model": "DeepSurv", "metric": "C-index", "value": cindex_deepsurv, "baseline_model": "Cox PH", "baseline_value": cindex_cox},
    ]
    metrics_df = pd.DataFrame(evaluation_rows)
    metrics_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

    histories = {
        "StressAutoencoder": ae_history,
        "AttentionLSTM": lstm_history,
        "CircadianAttentionLSTM": circ_history,
        "StressTCN": tcn_history,
        "DeepSurv": surv_history,
    }
    history_rows = []
    for model_name, history in histories.items():
        max_len = max((len(values) for values in history.values()), default=0)
        for idx in range(max_len):
            row = {"model": model_name, "epoch": idx + 1}
            for key, values in history.items():
                row[key] = values[idx] if idx < len(values) else np.nan
            history_rows.append(row)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    metrics_payload = {
        "run": {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQ_LEN,
            "stride": STRIDE,
            "source_rows": int(original_rows),
            "rows_used": int(len(df)),
            "ae_windows": int(len(ae_dataset)),
            "lstm_windows": int(len(lstm_dataset)),
            "survival_windows": int(len(surv_dataset)),
        },
        "metrics": [
            {key: _serializable_float(value) if key in {"value", "baseline_value"} else value for key, value in row.items()}
            for row in evaluation_rows
        ],
        "thresholds": {
            "ae_reconstruction_95p": _serializable_float(threshold),
            "flagged_windows": int(flags.sum()),
            "total_flag_windows": int(len(flags)),
            "flagged_pct": _serializable_float(100.0 * flags.sum() / max(len(flags), 1)),
            "mean_attrition_risk": _serializable_float(mean_risk),
        },
        "training": {
            model_name: {key: [_serializable_float(item) for item in values] for key, values in history.items()}
            for model_name, history in histories.items()
        },
    }
    (OUTPUT_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print("  [Saved] evaluation_metrics.csv, evaluation_metrics.json, training_history.csv, dataset_summary.csv")

    # ── Research Design: theory, causality, validation, parsimony ─────────────
    print("\n[RESEARCH DESIGN] Building theory, causality, robustness, and policy summary …")
    model_registry = {
        'StressMLP baseline': StressMLP(input_dim=len(FEATURE_COLS), hidden_dims=[32, 16]),
        'StressAutoencoder': trained_ae,
        'AttentionLSTM': trained_lstm,
        'CircadianAttentionLSTM': trained_circ,
        'StressTCN': trained_tcn,
        'DeepSurv': trained_surv,
    }
    research_summary = build_research_design_summary(
        df=df,
        feature_cols=FEATURE_COLS,
        models=model_registry,
        n_model_samples=len(df),
    )
    research_text = format_research_design_summary(research_summary)
    print(research_text)
    research_path = OUTPUT_DIR / 'research_design_summary.txt'
    research_path.write_text(research_text, encoding='utf-8')
    print("  [Saved] research_design_summary.txt")

    # ── Threshold Report ──────────────────────────────────────────────────────
    print("\n[REPORT] Generating threshold report …")
    reporter = ThresholdReportGenerator()
    report   = reporter.build_subject_report(
        subject_id   = df['subject_id'].iloc[0] if 'subject_id' in df.columns else 'unknown',
        ae_errors    = errors,
        ae_threshold = threshold,
        imbalance    = float(imbal.mean().item()),
        risk_score   = mean_risk,
        policy_implications = research_summary.managerial_implications,
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
