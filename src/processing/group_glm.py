"""Group-level random-effects GLM for multi-subject fNIRS analysis."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as t_dist

from processing.glm_analysis import GLMResult


@dataclass
class GroupGLMResult:
    """Second-level group statistics across subjects."""
    group_t_stat: np.ndarray       # (n_conditions, n_pairs)
    group_p_value: np.ndarray      # (n_conditions, n_pairs)
    subject_betas: np.ndarray      # (n_subjects, n_conditions, n_pairs)
    condition_names: list[str]
    pair_labels: list[str]
    n_subjects: int


def run_group_glm(subject_results: list[GLMResult]) -> GroupGLMResult:
    """Run second-level random-effects analysis using summary statistics.

    For each (condition, pair):
      1. Collect first-level beta weights across subjects.
      2. One-sample t-test against zero (H0: population mean beta == 0).

    Parameters
    ----------
    subject_results : list of GLMResult
        First-level GLM results, one per subject. All must share the same
        condition names and pair labels.

    Returns
    -------
    GroupGLMResult with group-level t-statistics and p-values.
    """
    if len(subject_results) < 2:
        raise ValueError("Group analysis requires at least 2 subjects.")

    ref = subject_results[0]
    n_conditions = len(ref.contrast_names)
    n_pairs = len(ref.pair_labels)
    n_subjects = len(subject_results)

    # Validate consistent dimensions.
    for i, r in enumerate(subject_results):
        if r.beta.shape != (n_conditions, n_pairs):
            raise ValueError(
                f"Subject {i} has shape {r.beta.shape}, "
                f"expected ({n_conditions}, {n_pairs})."
            )

    # Stack betas: (n_subjects, n_conditions, n_pairs).
    betas = np.array([r.beta for r in subject_results])

    # One-sample t-test per (condition, pair).
    mean_beta = betas.mean(axis=0)
    std_beta = betas.std(axis=0, ddof=1)
    se = std_beta / np.sqrt(n_subjects)

    # Guard against zero variance.
    se_safe = np.maximum(se, 1e-15)
    group_t = mean_beta / se_safe

    df = n_subjects - 1
    group_p = 2 * (1 - t_dist.cdf(np.abs(group_t), df))

    return GroupGLMResult(
        group_t_stat=group_t,
        group_p_value=group_p,
        subject_betas=betas,
        condition_names=list(ref.contrast_names),
        pair_labels=list(ref.pair_labels),
        n_subjects=n_subjects,
    )
