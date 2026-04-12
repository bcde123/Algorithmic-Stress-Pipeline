"""
Visualizer — Algorithmic Pressure Visualization

Generates publication-ready plots that map physiological stress signatures
across the multi-modal space, supporting Pillars 3 and 4 of the analysis.

Key outputs:
  1. Latent Stress Regime timeline     — reconstruction error over time
  2. Psychological Signature heatmap   — cluster profiles
  3. Algorithmic Pressure node graph   — 2D projection of attention-weighted features
  4. Circadian stress curve            — 24h stress index + imbalance trajectory
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette — distinct, accessible colours per dataset
# ──────────────────────────────────────────────────────────────────────────────
DATASET_COLOURS = {
    'WESAD':          '#4C72B0',
    'InducedStress':  '#DD8452',
    'MMASH':          '#55A868',
    'SWELL':          '#C44E52',
    'Unknown':        '#8172B2',
}

CLUSTER_COLOURS = [
    '#e63946', '#2a9d8f', '#f4a261', '#457b9d', '#6d6875'
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save the figure to disk or show it interactively."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Latent Stress Regime Timeline
# ──────────────────────────────────────────────────────────────────────────────

def plot_regime_timeline(
    reconstruction_errors: np.ndarray,
    threshold: float,
    flags: np.ndarray,
    title: str = "Latent Stress Regime Detection",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot reconstruction error over time, highlighting regime flags.

    Args:
        reconstruction_errors: 1D array of per-timestep MSE values.
        threshold:             Calibration baseline threshold (95th pct).
        flags:                 Binary array — 1 = Latent Stress Regime.
        title:                 Plot title.
        save_path:             If provided, saves to file instead of showing.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(len(reconstruction_errors))

    ax.plot(t, reconstruction_errors, lw=1.2, color='#4C72B0', label='Reconstruction Error')
    ax.axhline(threshold, color='#C44E52', ls='--', lw=1.5, label=f'Threshold ({threshold:.4f})')

    # Shade detected regime windows
    in_regime = False
    start = 0
    for i, f in enumerate(flags):
        if f == 1 and not in_regime:
            start = i; in_regime = True
        elif f == 0 and in_regime:
            ax.axvspan(start, i, alpha=0.25, color='#C44E52', label='Latent Stress Regime' if start == 0 else '')
            in_regime = False
    if in_regime:
        ax.axvspan(start, len(flags), alpha=0.25, color='#C44E52')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Reconstruction MSE', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper right')
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Psychological Signature Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_signature_heatmap(
    cluster_centers: np.ndarray,
    feature_names: Sequence[str],
    state_labels: Optional[List[str]] = None,
    title: str = "Psychological Signature Profiles",
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap showing the mean physiological profile of each K-Means cluster.

    Args:
        cluster_centers: (n_clusters, n_features) array.
        feature_names:   Column names matching the last dimension.
        state_labels:    Optional list of psychological state names per cluster.
        save_path:       Path to save the figure.
    """
    n_clusters, n_features = cluster_centers.shape
    if state_labels is None:
        state_labels = [f'Cluster {i}' for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(max(8, n_features * 1.2), max(4, n_clusters * 1.0)))

    img = ax.imshow(cluster_centers, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Z-scored feature value', fontsize=10)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(state_labels, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Annotate cells
    for r in range(n_clusters):
        for c in range(n_features):
            val = cluster_centers[r, c]
            ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color='black')

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Algorithmic Pressure 2D Node Graph (PCA projection)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pressure_nodes(
    profiles: np.ndarray,
    clusters: np.ndarray,
    state_labels: Optional[List[str]] = None,
    title: str = "Algorithmic Pressure — Attention-Weighted Feature Space",
    save_path: Optional[str] = None,
) -> None:
    """
    Two-dimensional PCA scatter plot of attention-weighted profiles,
    coloured by cluster (psychological signature cluster).

    Args:
        profiles:     (N, n_features) array of attention-weighted feature vectors.
        clusters:     (N,) integer cluster assignment per sample.
        state_labels: Optional list of cluster state names (for the legend).
        save_path:    Path to save figure.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(profiles)

    n_clusters = int(clusters.max()) + 1
    if state_labels is None:
        state_labels = [f'Cluster {i}' for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(9, 7))

    for k in range(n_clusters):
        mask = clusters == k
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=30, alpha=0.7,
            color=CLUSTER_COLOURS[k % len(CLUSTER_COLOURS)],
            label=state_labels[k],
            edgecolors='none',
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, markerscale=1.5)
    ax.grid(True, lw=0.4, alpha=0.5)

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Circadian Stress & Imbalance Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_circadian_curve(
    stress_index: np.ndarray,
    imbalance: np.ndarray,
    attn_weights: Optional[np.ndarray] = None,
    title: str = "24-Hour Stress Dynamics",
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel plot showing the circadian stress index and cumulative imbalance,
    with optional attention weight overlay.

    Args:
        stress_index: (T,) array of instantaneous stress index values [0, 1].
        imbalance:    (T,) array of cumulative imbalance scores.
        attn_weights: Optional (T,) array of attention weights to overlay.
        title:        Plot title.
        save_path:    Path to save figure.
    """
    T = len(stress_index)
    hours = np.linspace(0, 24, T)

    n_panels = 3 if attn_weights is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    if n_panels == 2:
        axes = list(axes)

    # Panel 1 — Stress Index
    axes[0].fill_between(hours, stress_index, alpha=0.4, color='#C44E52')
    axes[0].plot(hours, stress_index, lw=1.5, color='#C44E52')
    axes[0].set_ylabel('Stress Index', fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(title, fontsize=13, fontweight='bold')
    axes[0].grid(True, lw=0.4, alpha=0.5)

    # Panel 2 — Cumulative Imbalance
    axes[1].fill_between(hours, imbalance, alpha=0.4, color='#DD8452')
    axes[1].plot(hours, imbalance, lw=1.5, color='#DD8452')
    axes[1].set_ylabel('Cumulative Imbalance', fontsize=11)
    axes[1].grid(True, lw=0.4, alpha=0.5)

    # Panel 3 (optional) — Attention Weights
    if attn_weights is not None:
        axes[2].plot(hours, attn_weights, lw=1.2, color='#4C72B0')
        axes[2].set_ylabel('Attention Weight', fontsize=11)
        axes[2].set_xlabel('Hour of Day', fontsize=11)
        axes[2].grid(True, lw=0.4, alpha=0.5)
    else:
        axes[-1].set_xlabel('Hour of Day', fontsize=11)

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cross-dataset Stress Distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_stress_distribution(
    combined_df: pd.DataFrame,
    dataset_col: str = 'dataset',
    stress_col: str = 'stress_index',
    title: str = "Stress Index Distribution by Dataset",
    save_path: Optional[str] = None,
) -> None:
    """
    Overlapping KDE-style histogram of stress_index for each dataset.

    Args:
        combined_df: Combined DataFrame with dataset and stress_index columns.
        dataset_col: Name of the dataset identifier column.
        stress_col:  Name of the stress index column.
        save_path:   Path to save figure.
    """
    import warnings
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        gaussian_kde = None

    datasets = combined_df[dataset_col].unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    for ds in sorted(datasets):
        subset = combined_df.loc[combined_df[dataset_col] == ds, stress_col].dropna().values
        if len(subset) < 2:
            continue
        colour = DATASET_COLOURS.get(ds, DATASET_COLOURS['Unknown'])
        if gaussian_kde is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                x_grid = np.linspace(0, 1, 300)
                try:
                    kde = gaussian_kde(subset, bw_method=0.15)
                    ax.fill_between(x_grid, kde(x_grid), alpha=0.35, color=colour, label=ds)
                    ax.plot(x_grid, kde(x_grid), lw=1.5, color=colour)
                except Exception:
                    ax.hist(subset, bins=30, density=True, alpha=0.5, color=colour, label=ds)
        else:
            ax.hist(subset, bins=30, density=True, alpha=0.5, color=colour, label=ds)

    ax.set_xlabel('Unified Stress Index', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.4, alpha=0.5)

    _save_or_show(fig, save_path)
