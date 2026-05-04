"""
report_generator.py — Threshold-to-Report Output Module

Translates raw model outputs (reconstruction errors, imbalance scores, survival
risk scores) into human-readable per-subject threshold reports.

Example output
──────────────
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PHYSIOLOGICAL STRESS THRESHOLD REPORT                                   │
  │ Generated: 2026-04-12 02:00:00                                          │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ Subject: mock_01                                                         │
  │   • Latent Stress Regime: 14 / 80 windows exceeded threshold (17.5%)   │
  │   • Consecutive Breach:   3 sessions (⚠ SUSTAINED — burnout risk)      │
  │   • Cumulative Imbalance: 4.72 (Softplus units)                         │
  │   • DeepSurv Risk Score:  0.834  →  HIGH ATTRITION RISK                │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubjectReport:
    """Holds all model-output summaries for one subject."""
    subject_id:          str

    # Pillar 3 — Autoencoder
    ae_errors:           Optional[np.ndarray] = None   # per-window reconstruction errors
    ae_threshold:        Optional[float]       = None   # 95th-percentile calibration value
    ae_flags:            Optional[np.ndarray]  = None   # binary 0/1 per window

    # Pillar 1b — CircadianAttentionLSTM
    imbalance_score:     Optional[float]       = None   # scalar Softplus output (cumulative)

    # Pillar 5 — DeepSurv
    risk_score:          Optional[float]       = None   # log-hazard scalar

    # Consecutive-session tracking (populated by ThresholdReportGenerator)
    consecutive_breaches: int                  = 0

    # Research-design layer — policy and managerial implications
    policy_implications: List[str]             = field(default_factory=list)


@dataclass
class ThresholdConfig:
    """Configurable thresholds for the report narative."""
    ae_breach_pct_warn:     float = 0.10   # ≥ 10 % windows in regime → warn
    ae_breach_pct_critical: float = 0.25   # ≥ 25 % windows          → critical
    consecutive_warn:       int   = 2       # ≥ 2 consecutive sessions → sustained
    imbalance_warn:         float = 3.0    # Softplus imbalance units
    risk_warn:              float = 0.5    # log-hazard ≥ 0.5 → elevated
    risk_critical:          float = 0.8    # log-hazard ≥ 0.8 → high


# ─────────────────────────────────────────────────────────────────────────────
# ThresholdReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdReportGenerator:
    """
    Converts raw model outputs into structured, human-readable risk reports.

    Usage
    ─────
    >>> gen = ThresholdReportGenerator()
    >>> rpt = gen.build_subject_report(
    ...     subject_id   = 'S2',
    ...     ae_errors    = errors_array,
    ...     ae_threshold = threshold_value,
    ...     imbalance    = 4.2,
    ...     risk_score   = 0.78,
    ... )
    >>> print(gen.format_report(rpt))
    >>> gen.save_report(rpt, path='outputs/threshold_report.txt')
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()

    # ------------------------------------------------------------------
    def build_subject_report(
        self,
        subject_id:   str,
        ae_errors:    Optional[np.ndarray] = None,
        ae_threshold: Optional[float]       = None,
        imbalance:    Optional[float]       = None,
        risk_score:   Optional[float]       = None,
        policy_implications: Optional[List[str]] = None,
    ) -> SubjectReport:
        """Construct a SubjectReport from model outputs."""
        flags = None
        if ae_errors is not None and ae_threshold is not None:
            flags = (np.asarray(ae_errors) > ae_threshold).astype(int)

        rpt = SubjectReport(
            subject_id    = subject_id,
            ae_errors     = np.asarray(ae_errors)  if ae_errors    is not None else None,
            ae_threshold  = ae_threshold,
            ae_flags      = flags,
            imbalance_score = imbalance,
            risk_score      = risk_score,
            policy_implications = policy_implications or [],
        )
        return rpt

    # ------------------------------------------------------------------
    def format_report(self, rpt: SubjectReport) -> str:
        """Render a SubjectReport as a human-readable string."""
        cfg  = self.config
        now  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines: List[str] = []

        W = 74
        lines.append('─' * W)
        lines.append(f"  PHYSIOLOGICAL STRESS THRESHOLD REPORT")
        lines.append(f"  Generated: {now}")
        lines.append('─' * W)
        lines.append(f"  Subject: {rpt.subject_id}")
        lines.append('')

        # ── Pillar 3: Autoencoder ────────────────────────────────────────────
        if rpt.ae_flags is not None and rpt.ae_threshold is not None:
            n_total   = len(rpt.ae_flags)
            n_flagged = int(rpt.ae_flags.sum())
            pct       = n_flagged / max(n_total, 1)
            pct_str   = f"{pct * 100:.1f}%"

            if pct >= cfg.ae_breach_pct_critical:
                severity = "🔴 CRITICAL"
            elif pct >= cfg.ae_breach_pct_warn:
                severity = "🟡 ELEVATED"
            else:
                severity = "🟢 NORMAL"

            lines.append(f"  [Pillar 3] Latent Stress Regime Detection")
            lines.append(f"    • Windows flagged:    {n_flagged} / {n_total}  ({pct_str})  {severity}")
            lines.append(f"    • AE threshold (95p): {rpt.ae_threshold:.5f}")

            # Consecutive breach run
            if rpt.consecutive_breaches >= cfg.consecutive_warn:
                lines.append(
                    f"    ⚠ SUSTAINED BREACH: {rpt.consecutive_breaches} consecutive "
                    f"sessions above threshold — elevated burnout risk."
                )
            lines.append('')

        # ── Pillar 1b: Imbalance ──────────────────────────────────────────────
        if rpt.imbalance_score is not None:
            imb = rpt.imbalance_score
            tag = "🔴 HIGH"  if imb >= cfg.imbalance_warn * 1.5 else \
                  "🟡 MODERATE" if imb >= cfg.imbalance_warn else "🟢 LOW"
            lines.append(f"  [Pillar 1b] Cumulative Work-Life Imbalance")
            lines.append(f"    • Imbalance score:    {imb:.4f}  ({tag})")
            lines.append('')

        # ── Pillar 5: Survival Risk ───────────────────────────────────────────
        if rpt.risk_score is not None:
            rs = rpt.risk_score
            if rs >= cfg.risk_critical:
                risk_tag = "🔴 HIGH ATTRITION RISK"
            elif rs >= cfg.risk_warn:
                risk_tag = "🟡 ELEVATED ATTRITION RISK"
            else:
                risk_tag = "🟢 LOW ATTRITION RISK"
            lines.append(f"  [Pillar 5] DeepSurv Attrition Risk")
            lines.append(f"    • Log-hazard score:   {rs:.4f}  →  {risk_tag}")
            lines.append('')

        if rpt.policy_implications:
            lines.append(f"  Managerial and Policy Implications")
            for implication in rpt.policy_implications[:4]:
                lines.append(f"    • {implication}")
            lines.append('')

        lines.append('─' * W)
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def build_multi_subject_report(
        self,
        reports: List[SubjectReport],
    ) -> str:
        """Concatenate formatted reports for multiple subjects."""
        sections = [self.format_report(r) for r in reports]
        return '\n\n'.join(sections)

    # ------------------------------------------------------------------
    def save_report(
        self,
        report: 'str | SubjectReport | List[SubjectReport]',
        path: str,
    ) -> None:
        """Write the formatted report to a text file."""
        if isinstance(report, str):
            text = report
        elif isinstance(report, list):
            text = self.build_multi_subject_report(report)
        else:
            text = self.format_report(report)

        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_consecutive_breaches(flags: np.ndarray) -> int:
        """
        Returns the length of the longest consecutive run of 1s in ``flags``.
        Used to detect sustained threshold exceedances across sessions.
        """
        max_run = cur_run = 0
        for f in flags:
            if f:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        return max_run
