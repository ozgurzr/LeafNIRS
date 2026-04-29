"""GLM statistical analysis for fNIRS concentration data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist


@dataclass
class GLMResult:
    """Results from GLM analysis."""
    beta: np.ndarray           # (n_conditions, n_pairs) — regression weights
    t_stat: np.ndarray         # (n_conditions, n_pairs) — t-statistics
    p_value: np.ndarray        # (n_conditions, n_pairs) — p-values (two-tailed)
    contrast_names: list[str]  # condition labels
    pair_labels: list[str]     # S-D pair labels
    design_matrix: np.ndarray  # (n_time, n_regressors)
    residuals: np.ndarray      # (n_time, n_pairs)
    df: int                    # degrees of freedom


def canonical_hrf(t: np.ndarray, peak: float = 5.0, undershoot: float = 15.0,
                  peak_width: float = 1.0, undershoot_width: float = 1.0,
                  ratio: float = 6.0) -> np.ndarray:
    """Double-gamma canonical hemodynamic response function (SPM model).

    Parameters
    ----------
    t : array — time vector in seconds (must start at 0 or positive)
    peak : float — time of peak response (~5s)
    undershoot : float — time of undershoot (~15s)
    peak_width : float — width parameter for peak gamma
    undershoot_width : float — width parameter for undershoot gamma
    ratio : float — peak-to-undershoot amplitude ratio

    Returns
    -------
    hrf : array — normalized HRF values at each time point
    """
    from scipy.stats import gamma as gamma_dist

    t = np.maximum(t, 0.0)
    a1 = peak / peak_width
    a2 = undershoot / undershoot_width

    h = (gamma_dist.pdf(t, a1, scale=peak_width) -
         gamma_dist.pdf(t, a2, scale=undershoot_width) / ratio)

    # Normalize to unit peak.
    peak_val = np.max(np.abs(h))
    if peak_val > 0:
        h = h / peak_val
    return h


def build_design_matrix(time: np.ndarray, stimuli: list,
                        fs: float, hrf_duration: float = 32.0,
                        add_drift: int = 1) -> tuple[np.ndarray, list[str]]:
    """Build GLM design matrix by convolving stimulus onsets with canonical HRF.

    Parameters
    ----------
    time : array — recording time vector
    stimuli : list of StimulusInfo — stimulus conditions with onset times
    fs : float — sampling rate in Hz
    hrf_duration : float — duration of HRF kernel in seconds
    add_drift : int — polynomial drift order (0=constant only, 1=linear, 2=quadratic)

    Returns
    -------
    X : array — (n_time, n_regressors) design matrix
    names : list — regressor names
    """
    n_time = len(time)
    t_hrf = np.arange(0, hrf_duration, 1.0 / fs)
    hrf = canonical_hrf(t_hrf)

    regressors = []
    names = []

    for stim in stimuli:
        onset_signal = np.zeros(n_time)
        for onset_time in stim.onset:
            idx = np.argmin(np.abs(time - onset_time))
            if 0 <= idx < n_time:
                onset_signal[idx] = stim.amplitude[np.argmin(np.abs(stim.onset - onset_time))] \
                    if hasattr(stim, 'amplitude') and stim.amplitude is not None else 1.0

        convolved = np.convolve(onset_signal, hrf)[:n_time]
        regressors.append(convolved)
        names.append(stim.name)

    # Polynomial drift regressors.
    t_norm = np.linspace(-1, 1, n_time)
    for order in range(add_drift + 1):
        if order == 0:
            regressors.append(np.ones(n_time))
            names.append("constant")
        else:
            regressors.append(t_norm ** order)
            names.append(f"drift_order{order}")

    X = np.column_stack(regressors)
    return X, names


def run_glm(data: np.ndarray, design_matrix: np.ndarray,
            condition_indices: list[int],
            condition_names: list[str],
            pair_labels: list[str]) -> GLMResult:
    """Fit GLM and compute t-statistics for each condition.

    Parameters
    ----------
    data : array — (n_time, n_pairs) concentration data (HbO or HbR)
    design_matrix : array — (n_time, n_regressors) design matrix
    condition_indices : list — column indices in design_matrix that are stimulus conditions
    condition_names : list — names of the conditions
    pair_labels : list — S-D pair labels

    Returns
    -------
    GLMResult with beta weights, t-statistics, and p-values
    """
    X = design_matrix
    Y = data
    n_time, n_pairs = Y.shape
    n_regressors = X.shape[1]
    n_conditions = len(condition_indices)

    # OLS: beta = (X'X)^{-1} X'Y
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta_all = XtX_inv @ X.T @ Y
    residuals = Y - X @ beta_all
    df = n_time - n_regressors

    sigma2 = np.sum(residuals ** 2, axis=0) / df

    # Standard error: SE(beta_j) = sqrt(sigma^2 * (X'X)^{-1}_{jj}).
    beta = np.zeros((n_conditions, n_pairs))
    t_stat = np.zeros((n_conditions, n_pairs))
    p_value = np.ones((n_conditions, n_pairs))

    for ci, col_idx in enumerate(condition_indices):
        beta[ci, :] = beta_all[col_idx, :]
        se = np.sqrt(sigma2 * XtX_inv[col_idx, col_idx])
        se = np.maximum(se, 1e-15)  # avoid division by zero
        t_stat[ci, :] = beta[ci, :] / se
        p_value[ci, :] = 2 * (1 - t_dist.cdf(np.abs(t_stat[ci, :]), df))

    return GLMResult(
        beta=beta,
        t_stat=t_stat,
        p_value=p_value,
        contrast_names=condition_names,
        pair_labels=pair_labels,
        design_matrix=X,
        residuals=residuals,
        df=df,
    )


def compute_glm(hbo: np.ndarray, hbr: np.ndarray,
                time: np.ndarray, stimuli: list,
                pair_labels: list[str],
                fs: float) -> tuple[GLMResult, GLMResult]:
    """Run full GLM analysis on HbO and HbR data.

    Returns
    -------
    (glm_hbo, glm_hbr) — GLMResult for each chromophore
    """
    X, reg_names = build_design_matrix(time, stimuli, fs)

    condition_indices = []
    condition_names = []
    for i, name in enumerate(reg_names):
        if name not in ("constant",) and not name.startswith("drift_"):
            condition_indices.append(i)
            condition_names.append(name)

    glm_hbo = run_glm(hbo, X, condition_indices, condition_names, pair_labels)
    glm_hbr = run_glm(hbr, X, condition_indices, condition_names, pair_labels)

    return glm_hbo, glm_hbr
