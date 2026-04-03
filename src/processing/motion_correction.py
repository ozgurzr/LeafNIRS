"""Motion artifact detection and correction for fNIRS OD signals.

Implements TDDR (Fishburn et al., 2019) and cubic spline correction.
Applied to OD data before bandpass filtering.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def detect_artifacts(
    od: np.ndarray,
    fs: float,
    threshold: float = 3.5,
    win_sec: float = 1.0,
    merge_sec: float = 0.5,
) -> np.ndarray:
    """Detect motion artifacts using temporal derivative + MAD threshold.

    Parameters
    ----------
    od : ndarray, shape (n_time, n_ch)
    fs : float
        Sampling rate (Hz).
    threshold : float
        MAD multiplier for outlier detection.
    win_sec : float
        Expand each artifact by ±win_sec seconds.
    merge_sec : float
        Merge artifact segments closer than this.

    Returns
    -------
    mask : ndarray, shape (n_time, n_ch), dtype bool
    """
    od = np.asarray(od, dtype=np.float64)
    n_time, n_ch = od.shape
    expand = max(int(win_sec * fs / 2), 1)
    merge_gap = max(int(merge_sec * fs), 1)

    mask = np.zeros((n_time, n_ch), dtype=bool)

    for ch in range(n_ch):
        signal = od[:, ch]
        deriv = np.diff(signal, prepend=signal[0])
        abs_deriv = np.abs(deriv)

        median_d = np.median(abs_deriv)
        mad = np.median(np.abs(abs_deriv - median_d))

        if mad < 1e-12:
            std_d = np.std(abs_deriv)
            if std_d < 1e-12:
                continue
            channel_mask = abs_deriv > threshold * std_d
        else:
            # 1.4826 makes MAD consistent with std for normal distributions
            channel_mask = abs_deriv > median_d + threshold * 1.4826 * mad

        if np.any(channel_mask):
            indices = np.where(channel_mask)[0]
            for idx in indices:
                start = max(0, idx - expand)
                end = min(n_time, idx + expand + 1)
                channel_mask[start:end] = True

        if merge_gap > 0 and np.any(channel_mask):
            indices = np.where(channel_mask)[0]
            if len(indices) > 1:
                gaps = np.diff(indices)
                for i, gap in enumerate(gaps):
                    if gap <= merge_gap:
                        channel_mask[indices[i]:indices[i + 1] + 1] = True

        mask[:, ch] = channel_mask

    return mask


def correct_tddr(od: np.ndarray) -> np.ndarray:
    """Temporal Derivative Distribution Repair (Fishburn et al., 2019).

    Uses IRLS with Tukey's biweight to robustly estimate clean derivatives,
    then integrates back to produce corrected signal.

    Parameters
    ----------
    od : ndarray, shape (n_time, n_ch)

    Returns
    -------
    corrected : ndarray, shape (n_time, n_ch)
    """
    od = np.asarray(od, dtype=np.float64)
    n_time, n_ch = od.shape
    corrected = np.zeros_like(od)

    for ch in range(n_ch):
        signal = od[:, ch]
        deriv = np.diff(signal)
        clean_deriv = _robust_derivative(deriv)
        corrected[0, ch] = signal[0]
        corrected[1:, ch] = signal[0] + np.cumsum(clean_deriv)

    return corrected


def _robust_derivative(deriv: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """IRLS with Tukey's biweight to downweight outlier derivatives."""
    n = len(deriv)
    weights = np.ones(n)
    k = 4.685  # Tukey's biweight tuning constant

    for _ in range(max_iter):
        w_sum = np.sum(weights)
        if w_sum < 1e-12:
            break
        mu = np.sum(weights * deriv) / w_sum

        residuals = deriv - mu
        mad = np.median(np.abs(residuals))
        if mad < 1e-12:
            break

        u = residuals / (k * 1.4826 * mad)
        new_weights = np.where(np.abs(u) <= 1.0, (1 - u**2)**2, 0.0)

        if np.max(np.abs(new_weights - weights)) < 1e-6:
            weights = new_weights
            break
        weights = new_weights

    w_sum = np.sum(weights)
    if w_sum < 1e-12:
        return deriv

    mu = np.sum(weights * deriv) / w_sum
    clean = deriv.copy()
    clean[weights < 0.01] = mu
    return clean


def correct_spline(
    od: np.ndarray,
    artifact_mask: np.ndarray,
    pad_samples: int = 5,
) -> np.ndarray:
    """Replace artifact segments with cubic spline interpolation.

    Parameters
    ----------
    od : ndarray, shape (n_time, n_ch)
    artifact_mask : ndarray, shape (n_time, n_ch), dtype bool
    pad_samples : int
        Clean samples on each side used as spline anchors.

    Returns
    -------
    corrected : ndarray, shape (n_time, n_ch)
    """
    od = np.asarray(od, dtype=np.float64)
    artifact_mask = np.asarray(artifact_mask, dtype=bool)
    n_time, n_ch = od.shape
    corrected = od.copy()

    for ch in range(n_ch):
        ch_mask = artifact_mask[:, ch]
        if not np.any(ch_mask):
            continue

        clean_idx = np.where(~ch_mask)[0]
        artifact_idx = np.where(ch_mask)[0]

        if len(clean_idx) < 4:
            continue

        try:
            cs = CubicSpline(clean_idx, od[clean_idx, ch])
            corrected[artifact_idx, ch] = cs(artifact_idx)
        except Exception:
            corrected[artifact_idx, ch] = np.interp(
                artifact_idx, clean_idx, od[clean_idx, ch],
            )

    return corrected
