"""
research_design.py — Theory, Causality, Validation, and Parsimony Layer

This module turns model outputs and harmonized physiological streams into
research-design evidence that is easier to align with organizational behavior,
labor economics, and management scholarship. It does not claim causality from
prediction alone; causal estimates are reported with explicit adjustment sets,
identification assumptions, and intervention-contrast checks when the datasets
contain experimentally induced or naturally occurring work-design labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score


@dataclass
class TheoryConstruct:
    name: str
    framework: str
    operationalization: str
    interpretation: str
    literature_alignment: str


@dataclass
class ComplexityResult:
    model_name: str
    trainable_parameters: int
    non_trainable_parameters: int
    sample_to_parameter_ratio: float
    status: str


@dataclass
class CausalEstimate:
    treatment: str
    outcome: str
    estimand: str
    effect: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    adjustment_set: List[str]
    assumptions: List[str]
    method: str
    status: str


@dataclass
class InterventionValidationResult:
    contrast: str
    outcome: str
    control_mean: float
    treatment_mean: float
    effect: float
    ci_low: float
    ci_high: float
    n_control: int
    n_treatment: int
    source: str
    status: str


@dataclass
class RobustnessResult:
    check: str
    metric: str
    value: float
    status: str


@dataclass
class ExternalValidationResult:
    holdout_dataset: str
    metric: str
    value: float
    n_holdout: int
    status: str


@dataclass
class ResearchDesignSummary:
    constructs: List[TheoryConstruct]
    complexity: List[ComplexityResult]
    causal_estimates: List[CausalEstimate]
    intervention_validation: List[InterventionValidationResult]
    robustness: List[RobustnessResult]
    external_validation: List[ExternalValidationResult]
    managerial_implications: List[str]


def build_theoretical_framework() -> List[TheoryConstruct]:
    return [
        TheoryConstruct(
            name="Algorithmic demand",
            framework="Job Demands-Resources model and labor-process economics",
            operationalization="High stress_index periods, EDA elevation, HR acceleration, and time-pressure/interruption labels.",
            interpretation="Represents workload intensity, monitoring pressure, task fragmentation, and constrained pacing imposed by algorithmic systems.",
            literature_alignment="Connects physiological stress prediction to job demands, work intensification, and managerial control rather than treating stress as a purely technical label.",
        ),
        TheoryConstruct(
            name="Recovery resources and autonomy",
            framework="Demand-Control model and conservation-of-resources theory",
            operationalization="HRV recovery, low EDA regimes, rest/recovery interventions, and circadian imbalance outputs.",
            interpretation="Captures whether workers have sufficient recovery slack and control to return toward physiological baseline after demand shocks.",
            literature_alignment="Links wearable signals to managerial levers such as break scheduling, task rotation, and autonomy over pacing.",
        ),
        TheoryConstruct(
            name="Physiological strain",
            framework="Effort-Recovery theory and occupational health economics",
            operationalization="Sustained reconstruction-error breaches, lowered HRV, elevated HR, and cumulative imbalance scores.",
            interpretation="Measures the accumulated cost of repeated demand exposure before it appears as absenteeism, burnout, or attrition.",
            literature_alignment="Frames stress as a leading indicator of human-capital depreciation and reduced worker welfare.",
        ),
        TheoryConstruct(
            name="Exit and productivity risk",
            framework="Labor supply, turnover, and compensating-differentials literature",
            operationalization="DeepSurv/Cox hazard scores and intervention contrasts that quantify stress-risk changes under different work designs.",
            interpretation="Translates physiological strain into economically interpretable risks for retention, productivity, and managerial policy.",
            literature_alignment="Aligns the contribution with economics and management outcomes rather than model accuracy alone.",
        ),
    ]


def summarize_model_complexity(models: Dict[str, object], n_samples: int) -> List[ComplexityResult]:
    results: List[ComplexityResult] = []
    safe_n = max(int(n_samples), 1)
    for name, model in models.items():
        trainable = 0
        non_trainable = 0
        if hasattr(model, "parameters"):
            for param in model.parameters():
                count = int(param.numel())
                if getattr(param, "requires_grad", False):
                    trainable += count
                else:
                    non_trainable += count
        ratio = safe_n / max(trainable, 1)
        if ratio >= 10:
            status = "parsimonious"
        elif ratio >= 1:
            status = "acceptable for validation, monitor overfitting"
        else:
            status = "high complexity relative to available samples"
        results.append(
            ComplexityResult(
                model_name=name,
                trainable_parameters=trainable,
                non_trainable_parameters=non_trainable,
                sample_to_parameter_ratio=ratio,
                status=status,
            )
        )
    return sorted(results, key=lambda item: item.trainable_parameters)


def estimate_high_stress_effect(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "stress_index",
    threshold: float = 0.6,
    adjustment_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = ("dataset",),
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> CausalEstimate:
    if adjustment_cols is None:
        adjustment_cols = ["ACC_x", "ACC_y", "ACC_z", "TEMP"]

    required = [outcome_col, treatment_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return _empty_causal_result(outcome_col, threshold, adjustment_cols, f"missing columns: {missing}")

    use_cols = [outcome_col, treatment_col]
    use_cols += [col for col in adjustment_cols if col in df.columns]
    use_cols += [col for col in (categorical_cols or []) if col in df.columns]
    work = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 20:
        return _empty_causal_result(outcome_col, threshold, adjustment_cols, "not enough observations")

    work["high_stress_treatment"] = (work[treatment_col] >= threshold).astype(float)
    n_treated = int(work["high_stress_treatment"].sum())
    n_control = int(len(work) - n_treated)
    if n_treated < 5 or n_control < 5:
        return _empty_causal_result(outcome_col, threshold, adjustment_cols, "insufficient treatment/control overlap", n_treated, n_control)

    X = _design_matrix(work, "high_stress_treatment", adjustment_cols, categorical_cols)
    y = work[outcome_col].astype(float).to_numpy()
    model = LinearRegression().fit(X, y)
    effect = float(model.coef_[0])

    rng = np.random.default_rng(random_state)
    boot_effects: List[float] = []
    indices = np.arange(len(work))
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sample = work.iloc[sample_idx]
        if sample["high_stress_treatment"].nunique() < 2:
            continue
        X_b = _design_matrix(sample, "high_stress_treatment", adjustment_cols, categorical_cols)
        y_b = sample[outcome_col].astype(float).to_numpy()
        boot_effects.append(float(LinearRegression().fit(X_b, y_b).coef_[0]))

    if boot_effects:
        se = float(np.std(boot_effects, ddof=1)) if len(boot_effects) > 1 else 0.0
        ci_low, ci_high = np.percentile(boot_effects, [2.5, 97.5])
    else:
        se = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    status = "estimated under conditional exchangeability; validate with intervention contrasts"
    if np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low <= 0 <= ci_high):
        status = "direction uncertain; confidence interval crosses zero"

    return CausalEstimate(
        treatment=f"stress_index >= {threshold:.2f}",
        outcome=outcome_col,
        estimand="Average treatment effect of high algorithmic-stress exposure on the outcome",
        effect=effect,
        standard_error=se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_treated=n_treated,
        n_control=n_control,
        adjustment_set=[col for col in adjustment_cols if col in df.columns] + [col for col in (categorical_cols or []) if col in df.columns],
        assumptions=[
            "No unmeasured confounding after physical-exertion and dataset adjustment.",
            "Positivity: comparable high- and low-stress observations exist within the adjusted data.",
            "SUTVA: one worker-window's exposure does not alter another worker-window's potential outcome.",
        ],
        method="Adjusted linear regression with nonparametric bootstrap confidence interval",
        status=status,
    )


def validate_intervention_contrast(
    df: pd.DataFrame,
    outcome_col: str = "stress_index",
    intervention_col: str = "intervention",
    treatment_labels: Optional[Iterable[str]] = None,
    control_labels: Optional[Iterable[str]] = None,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> InterventionValidationResult:
    if treatment_labels is None:
        treatment_labels = {
            "stress",
            "time pressure",
            "interruption",
            "high_workload",
            "maximum load",
            "algorithmic_pressure",
        }
    if control_labels is None:
        control_labels = {"baseline", "recovery", "rest", "no stress", "standard_work", "meditation"}

    if intervention_col not in df.columns or outcome_col not in df.columns:
        return _empty_intervention_result(outcome_col, "missing intervention or outcome column")

    work = df[[intervention_col, outcome_col, "dataset"] if "dataset" in df.columns else [intervention_col, outcome_col]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return _empty_intervention_result(outcome_col, "no valid intervention observations")

    labels = work[intervention_col].astype(str).str.lower().str.strip()
    treatment_set = {str(label).lower().strip() for label in treatment_labels}
    control_set = {str(label).lower().strip() for label in control_labels}
    treated = work.loc[labels.isin(treatment_set), outcome_col].astype(float).to_numpy()
    control = work.loc[labels.isin(control_set), outcome_col].astype(float).to_numpy()

    if len(treated) < 5 or len(control) < 5:
        return _empty_intervention_result(outcome_col, "insufficient labeled treatment/control intervention contrast", len(control), len(treated))

    effect = float(treated.mean() - control.mean())
    rng = np.random.default_rng(random_state)
    boot_effects = []
    for _ in range(n_bootstrap):
        t = rng.choice(treated, size=len(treated), replace=True)
        c = rng.choice(control, size=len(control), replace=True)
        boot_effects.append(float(t.mean() - c.mean()))
    ci_low, ci_high = np.percentile(boot_effects, [2.5, 97.5])

    if "dataset" in work.columns:
        source_values = sorted(set(work["dataset"].astype(str)))
        source = ", ".join(source_values)
    else:
        source = "intervention labels"

    status = "intervention contrast supports directional stress response"
    if ci_low <= 0 <= ci_high:
        status = "intervention contrast inconclusive; confidence interval crosses zero"

    return InterventionValidationResult(
        contrast="high-demand intervention minus recovery/control",
        outcome=outcome_col,
        control_mean=float(control.mean()),
        treatment_mean=float(treated.mean()),
        effect=effect,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_control=int(len(control)),
        n_treatment=int(len(treated)),
        source=source,
        status=status,
    )


def run_threshold_robustness(
    df: pd.DataFrame,
    outcome_col: str = "HRV",
    thresholds: Sequence[float] = (0.5, 0.6, 0.7),
) -> List[RobustnessResult]:
    results: List[RobustnessResult] = []
    effects = []
    for threshold in thresholds:
        estimate = estimate_high_stress_effect(
            df,
            outcome_col=outcome_col,
            threshold=threshold,
            n_bootstrap=30,
        )
        effects.append(estimate.effect)
        results.append(
            RobustnessResult(
                check=f"high-stress threshold {threshold:.2f}",
                metric=f"ATE on {outcome_col}",
                value=estimate.effect,
                status=estimate.status,
            )
        )
    finite_effects = [effect for effect in effects if np.isfinite(effect)]
    if finite_effects:
        signs = {np.sign(effect) for effect in finite_effects if effect != 0}
        status = "stable sign across thresholds" if len(signs) <= 1 else "sign changes across thresholds"
        results.append(
            RobustnessResult(
                check="threshold sensitivity",
                metric="effect range",
                value=float(np.max(finite_effects) - np.min(finite_effects)),
                status=status,
            )
        )
    return results


def leave_one_dataset_out_validation(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "stress_index",
    min_train: int = 50,
    min_holdout: int = 20,
) -> List[ExternalValidationResult]:
    if "dataset" not in df.columns or target_col not in df.columns:
        return [ExternalValidationResult("not_available", "AUROC", float("nan"), 0, "dataset labels unavailable")]

    work_cols = list(feature_cols) + [target_col, "dataset"]
    work = df[work_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    datasets = sorted(work["dataset"].astype(str).unique())
    if len(datasets) < 2:
        return [ExternalValidationResult(datasets[0] if datasets else "not_available", "AUROC", float("nan"), len(work), "external validation requires at least two datasets")]

    results: List[ExternalValidationResult] = []
    for holdout in datasets:
        train = work[work["dataset"].astype(str) != holdout]
        test = work[work["dataset"].astype(str) == holdout]
        if len(train) < min_train or len(test) < min_holdout:
            results.append(ExternalValidationResult(holdout, "AUROC", float("nan"), len(test), "insufficient train/holdout size"))
            continue

        y_train = train[target_col].astype(float).to_numpy()
        y_test = test[target_col].astype(float).to_numpy()
        y_test_binary = (y_test >= 0.5).astype(int)
        model = Ridge(alpha=1.0)
        model.fit(train[list(feature_cols)].astype(float).to_numpy(), y_train)
        pred = np.clip(model.predict(test[list(feature_cols)].astype(float).to_numpy()), 0, 1)

        if len(np.unique(y_test_binary)) > 1:
            value = float(roc_auc_score(y_test_binary, pred))
            metric = "AUROC"
            status = "passes external holdout" if value >= 0.70 else "weak external holdout performance"
        else:
            value = float(mean_absolute_error(y_test, pred))
            metric = "MAE"
            status = "single-class holdout; reported MAE instead of AUROC"

        results.append(ExternalValidationResult(holdout, metric, value, int(len(test)), status))
    return results


def build_managerial_implications(summary: ResearchDesignSummary) -> List[str]:
    implications = [
        "Use stress thresholds as decision-support triggers for work redesign, not as punitive individual surveillance metrics.",
        "Prioritize task pacing, recovery breaks, and interruption reduction when high-demand contrasts increase stress_index or reduce HRV.",
        "Require external validation on a new site or dataset before deploying risk scores for staffing, scheduling, or retention decisions.",
        "Prefer parsimonious models unless complex models materially outperform transparent baselines on external holdouts.",
    ]

    negative_hrv = [
        estimate for estimate in summary.causal_estimates
        if estimate.outcome == "HRV" and np.isfinite(estimate.effect) and estimate.effect < 0
    ]
    if negative_hrv:
        implications.append("Treat sustained high-stress exposure as a recovery-resource deficit because adjusted estimates indicate lower HRV under high stress.")

    strong_interventions = [
        result for result in summary.intervention_validation
        if np.isfinite(result.effect) and result.effect > 0 and result.ci_low > 0
    ]
    if strong_interventions:
        implications.append("Use intervention contrasts to quantify expected stress reductions from moving workers out of high-demand regimes.")

    return implications


def build_research_design_summary(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    models: Optional[Dict[str, object]] = None,
    n_model_samples: Optional[int] = None,
) -> ResearchDesignSummary:
    complexity = summarize_model_complexity(models or {}, n_model_samples or len(df))
    causal_estimates = [
        estimate_high_stress_effect(df, outcome_col="HRV") if "HRV" in df.columns else _empty_causal_result("HRV", 0.6, [], "HRV unavailable"),
        estimate_high_stress_effect(df, outcome_col="HR") if "HR" in df.columns else _empty_causal_result("HR", 0.6, [], "HR unavailable"),
    ]
    intervention_validation = [validate_intervention_contrast(df, outcome_col="stress_index")]
    robustness = run_threshold_robustness(df, outcome_col="HRV") if "HRV" in df.columns else []
    external_validation = leave_one_dataset_out_validation(df, feature_cols)

    summary = ResearchDesignSummary(
        constructs=build_theoretical_framework(),
        complexity=complexity,
        causal_estimates=causal_estimates,
        intervention_validation=intervention_validation,
        robustness=robustness,
        external_validation=external_validation,
        managerial_implications=[],
    )
    summary.managerial_implications = build_managerial_implications(summary)
    return summary


def format_research_design_summary(summary: ResearchDesignSummary) -> str:
    lines = ["RESEARCH DESIGN SUMMARY", "=" * 78]
    lines.append("Theoretical framework")
    for item in summary.constructs:
        lines.append(f"- {item.name} [{item.framework}]: {item.operationalization}")
    lines.append("")

    lines.append("Model parsimony")
    if summary.complexity:
        for item in summary.complexity:
            lines.append(
                f"- {item.model_name}: {item.trainable_parameters:,} trainable params; "
                f"sample/param={item.sample_to_parameter_ratio:.3f}; {item.status}"
            )
    else:
        lines.append("- No model objects supplied for parameter counting.")
    lines.append("")

    lines.append("Causal estimands")
    for item in summary.causal_estimates:
        lines.append(
            f"- {item.treatment} -> {item.outcome}: effect={item.effect:+.4f}, "
            f"95% CI [{item.ci_low:+.4f}, {item.ci_high:+.4f}], "
            f"n_treated={item.n_treated}, n_control={item.n_control}; {item.status}"
        )
    lines.append("")

    lines.append("Intervention validation")
    for item in summary.intervention_validation:
        lines.append(
            f"- {item.contrast} on {item.outcome}: effect={item.effect:+.4f}, "
            f"95% CI [{item.ci_low:+.4f}, {item.ci_high:+.4f}], source={item.source}; {item.status}"
        )
    lines.append("")

    lines.append("Robustness and external validation")
    for item in summary.robustness:
        lines.append(f"- Robustness {item.check}: {item.metric}={item.value:+.4f}; {item.status}")
    for item in summary.external_validation:
        lines.append(f"- External holdout {item.holdout_dataset}: {item.metric}={item.value:.4f}, n={item.n_holdout}; {item.status}")
    lines.append("")

    lines.append("Policy and managerial implications")
    for implication in summary.managerial_implications:
        lines.append(f"- {implication}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _design_matrix(
    df: pd.DataFrame,
    treatment_col: str,
    adjustment_cols: Sequence[str],
    categorical_cols: Optional[Sequence[str]],
) -> np.ndarray:
    cols = [treatment_col] + [col for col in adjustment_cols if col in df.columns]
    X = df[cols].astype(float).copy()
    for col in categorical_cols or []:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float)
            X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X.to_numpy(dtype=float)


def _empty_causal_result(
    outcome_col: str,
    threshold: float,
    adjustment_cols: Sequence[str],
    status: str,
    n_treated: int = 0,
    n_control: int = 0,
) -> CausalEstimate:
    return CausalEstimate(
        treatment=f"stress_index >= {threshold:.2f}",
        outcome=outcome_col,
        estimand="Average treatment effect of high algorithmic-stress exposure on the outcome",
        effect=float("nan"),
        standard_error=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_treated=n_treated,
        n_control=n_control,
        adjustment_set=list(adjustment_cols),
        assumptions=[],
        method="Adjusted linear regression with bootstrap confidence interval",
        status=status,
    )


def _empty_intervention_result(
    outcome_col: str,
    status: str,
    n_control: int = 0,
    n_treatment: int = 0,
) -> InterventionValidationResult:
    return InterventionValidationResult(
        contrast="high-demand intervention minus recovery/control",
        outcome=outcome_col,
        control_mean=float("nan"),
        treatment_mean=float("nan"),
        effect=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_control=n_control,
        n_treatment=n_treatment,
        source="not_available",
        status=status,
    )
