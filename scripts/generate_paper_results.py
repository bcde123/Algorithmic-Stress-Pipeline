from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PAPER_DIR = OUTPUT_DIR / "paper_figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "research_design_summary.txt"
REPORT_PATH = OUTPUT_DIR / "threshold_report.txt"
METRICS_CSV = OUTPUT_DIR / "evaluation_metrics.csv"
HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
DATASET_CSV = OUTPUT_DIR / "dataset_summary.csv"
METRICS_JSON = OUTPUT_DIR / "evaluation_metrics.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_research_summary(text: str):
    complexity = []
    for model, params, ratio, status in re.findall(
        r"- ([^:]+): ([\d,]+) trainable params; sample/param=([\d.]+); ([^\n]+)", text
    ):
        complexity.append({
            "model": model,
            "params": int(params.replace(",", "")),
            "sample_param": float(ratio),
            "status": status.strip(),
        })

    causal = []
    for threshold, outcome, effect, lo, hi, nt, nc, status in re.findall(
        r"stress_index >= ([\d.]+) -> ([A-Za-z_]+): effect=([+-]?[\d.]+), 95% CI \[([+-]?[\d.]+), ([+-]?[\d.]+)\], n_treated=(\d+), n_control=(\d+); ([^\n]+)",
        text,
    ):
        causal.append({
            "threshold": float(threshold),
            "outcome": outcome,
            "effect": float(effect),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_treated": int(nt),
            "n_control": int(nc),
            "status": status.strip(),
        })

    robustness = []
    for threshold, effect in re.findall(
        r"Robustness high-stress threshold ([\d.]+): ATE on HRV=([+-]?[\d.]+)", text
    ):
        robustness.append({"threshold": float(threshold), "effect": float(effect)})

    intervention = None
    match = re.search(
        r"high-demand intervention minus recovery/control on stress_index: effect=([+-]?[\d.]+), 95% CI \[([+-]?[\d.]+), ([+-]?[\d.]+)\], source=([^;]+); ([^\n]+)",
        text,
    )
    if match:
        intervention = {
            "effect": float(match.group(1)),
            "ci_low": float(match.group(2)),
            "ci_high": float(match.group(3)),
            "source": match.group(4),
            "status": match.group(5),
        }

    return complexity, causal, robustness, intervention


def parse_threshold_report(text: str):
    values = {}
    flagged = re.search(r"Windows flagged:\s+(\d+) / (\d+)\s+\(([\d.]+)%\)", text)
    threshold = re.search(r"AE threshold \(95p\):\s+([\d.]+)", text)
    imbalance = re.search(r"Imbalance score:\s+([+-]?[\d.]+)", text)
    risk = re.search(r"Log-hazard score:\s+([+-]?[\d.]+)", text)
    if flagged:
        values["flagged_windows"] = int(flagged.group(1))
        values["total_windows"] = int(flagged.group(2))
        values["flagged_pct"] = float(flagged.group(3))
    if threshold:
        values["ae_threshold"] = float(threshold.group(1))
    if imbalance:
        values["imbalance"] = float(imbalance.group(1))
    if risk:
        values["risk"] = float(risk.group(1))
    return values


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_metrics_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_model_parsimony(complexity):
    df = pd.DataFrame(complexity)
    if df.empty:
        return
    df = df.sort_values("params")
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(df))
    ax1.bar(x, df["params"], color="#4C72B0", alpha=0.85)
    ax1.set_ylabel("Trainable parameters", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["model"], rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, df["sample_param"], marker="o", color="#C44E52", linewidth=2)
    ax2.axhline(1.0, color="#8172B2", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Sample-to-parameter ratio", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_title("Model Parsimony and Sample Support")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "model_parsimony.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(PAPER_DIR / "model_parsimony.csv", index=False)


def save_causal_effects(causal, robustness, intervention):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    if causal:
        labels = [item["outcome"] for item in causal]
        effects = np.array([item["effect"] for item in causal])
        lows = np.array([item["ci_low"] for item in causal])
        highs = np.array([item["ci_high"] for item in causal])
        y = np.arange(len(labels))
        axes[0].errorbar(
            effects,
            y,
            xerr=[effects - lows, highs - effects],
            fmt="o",
            color="#4C72B0",
            ecolor="#4C72B0",
            capsize=4,
        )
        axes[0].axvline(0, color="black", linewidth=1)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels)
        axes[0].set_xlabel("Adjusted high-stress effect")
        axes[0].set_title("Causal Estimands with Bootstrap CI")
        axes[0].grid(axis="x", alpha=0.25)

    if robustness:
        thresholds = [item["threshold"] for item in robustness]
        effects = [item["effect"] for item in robustness]
        axes[1].plot(thresholds, effects, marker="o", linewidth=2, color="#C44E52")
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_xlabel("High-stress threshold")
        axes[1].set_ylabel("ATE on HRV")
        axes[1].set_title("Threshold Robustness")
        axes[1].grid(alpha=0.25)
        if intervention:
            txt = f"Intervention contrast\nΔ stress_index = {intervention['effect']:+.3f}\n95% CI [{intervention['ci_low']:+.3f}, {intervention['ci_high']:+.3f}]"
            axes[1].text(0.02, 0.04, txt, transform=axes[1].transAxes, fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "causal_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(causal).to_csv(PAPER_DIR / "causal_estimates.csv", index=False)
    pd.DataFrame(robustness).to_csv(PAPER_DIR / "robustness_estimates.csv", index=False)


def save_threshold_dashboard(values):
    labels = ["Regime windows", "Imbalance", "Attrition risk"]
    vals = [
        values.get("flagged_pct", 0.0) / 100.0,
        values.get("imbalance", 0.0) / 3.0,
        max(values.get("risk", 0.0), 0.0) / 0.8,
    ]
    colors = ["#55A868", "#DD8452", "#8172B2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals, color=colors, alpha=0.88)
    ax.axhline(1.0, color="#C44E52", linestyle="--", linewidth=1.2, label="warning threshold")
    ax.set_ylim(0, max(1.2, max(vals) + 0.2))
    ax.set_ylabel("Normalized risk indicator")
    ax.set_title("Per-Subject Threshold Dashboard")
    for i, val in enumerate(vals):
        ax.text(i, val + 0.03, f"{val:.2f}", ha="center", va="bottom")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "threshold_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([values]).to_csv(PAPER_DIR / "threshold_dashboard.csv", index=False)


def save_dataset_composition(dataset_df: pd.DataFrame):
    if dataset_df.empty:
        return
    df = dataset_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(df["dataset"], df["rows"], color="#4C72B0", alpha=0.88)
    axes[0].set_title("Rows Used by Dataset")
    axes[0].set_ylabel("Harmonized 1 Hz rows")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(df["dataset"], df["high_stress_rate"], color="#C44E52", alpha=0.88)
    axes[1].set_title("High-Stress Exposure Rate")
    axes[1].set_ylabel("Share with stress_index ≥ 0.60")
    axes[1].set_ylim(0, min(1.0, max(0.2, float(df["high_stress_rate"].max()) + 0.1)))
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "dataset_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    df = metrics_df.dropna(subset=["value"]).copy()
    if df.empty:
        return
    labels = [f"{row.model}\n{row.metric}" for row in df.itertuples()]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.18, df["value"], width=0.36, label="ASMP model", color="#4C72B0", alpha=0.88)
    if "baseline_value" in df.columns:
        baseline = pd.to_numeric(df["baseline_value"], errors="coerce")
        ax.bar(x + 0.18, baseline.fillna(0.0), width=0.36, label="Baseline", color="#DD8452", alpha=0.78)
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Evaluation Metrics Across Analytical Pillars")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "evaluation_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_df: pd.DataFrame):
    if history_df.empty or "epoch" not in history_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for model, group in history_df.groupby("model"):
        y_col = "val_loss" if "val_loss" in group and group["val_loss"].notna().any() else "train_loss"
        if y_col not in group:
            continue
        ax.plot(group["epoch"], group[y_col], marker="o", linewidth=2, label=f"{model} {y_col}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GPU Training Convergence by Model")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_results_dashboard(metrics_json: dict):
    if not metrics_json:
        return
    run = metrics_json.get("run", {})
    thresholds = metrics_json.get("thresholds", {})
    metrics = metrics_json.get("metrics", [])
    rows = [
        "ASMP GPU Results Dashboard",
        f"Device: {run.get('device', 'unknown')} | GPU: {run.get('gpu') or 'not available'}",
        f"Rows used: {run.get('rows_used', 'n/a')} / source rows: {run.get('source_rows', 'n/a')}",
        f"Windows: AE={run.get('ae_windows', 'n/a')}, LSTM={run.get('lstm_windows', 'n/a')}, Survival={run.get('survival_windows', 'n/a')}",
        f"AE 95p threshold={thresholds.get('ae_reconstruction_95p', 'n/a')} | flagged={thresholds.get('flagged_windows', 'n/a')}/{thresholds.get('total_flag_windows', 'n/a')}",
        "",
        "Key metrics",
    ]
    for item in metrics:
        value = item.get("value")
        baseline = item.get("baseline_value")
        value_txt = "nan" if value is None else f"{value:.4f}"
        baseline_txt = "" if baseline is None else f" | baseline={baseline:.4f}"
        rows.append(f"- {item.get('pillar')} {item.get('model')} {item.get('metric')}: {value_txt}{baseline_txt}")
    render_text_image("\n".join(rows), "gpu_results_screenshot.png", "Generated Results Screenshot", max_lines=36)


def render_text_image(text: str, filename: str, title: str, max_lines: int = 34):
    lines = [line[:115] for line in text.splitlines()[:max_lines]]
    fig_height = max(5, 0.27 * len(lines) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=14, fontweight="bold", va="top", family="sans-serif")
    ax.text(0.0, 0.92, "\n".join(lines), fontsize=9.5, va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    summary_text = read_text(SUMMARY_PATH)
    report_text = read_text(REPORT_PATH)
    metrics_df = read_csv(METRICS_CSV)
    history_df = read_csv(HISTORY_CSV)
    dataset_df = read_csv(DATASET_CSV)
    metrics_json = read_metrics_json(METRICS_JSON)
    complexity, causal, robustness, intervention = parse_research_summary(summary_text)
    threshold_values = parse_threshold_report(report_text)
    save_model_parsimony(complexity)
    save_causal_effects(causal, robustness, intervention)
    save_threshold_dashboard(threshold_values)
    save_dataset_composition(dataset_df)
    save_metric_comparison(metrics_df)
    save_training_curves(history_df)
    save_results_dashboard(metrics_json)
    render_text_image(summary_text, "research_design_screenshot.png", "Research Design Summary")
    render_text_image(report_text, "threshold_report_screenshot.png", "Generated Threshold Report", max_lines=24)
    print(f"Wrote paper figures to {PAPER_DIR}")


if __name__ == "__main__":
    main()
