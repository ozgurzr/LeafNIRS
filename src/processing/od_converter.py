"""Optical Density (OD) Converter: OD = -log10(I / I0)."""
from __future__ import annotations

import numpy as np

_EPSILON = 1e-10


def intensity_to_od(
    intensity: np.ndarray,
    baseline_start: int = 0,
    baseline_end: int | None = None,
) -> np.ndarray:
    """Convert raw intensity to optical density (OD).

    Parameters
    ----------
    intensity : ndarray, shape (n_time, n_ch)
    baseline_start, baseline_end : int
        Index range for baseline averaging. None = entire recording.

    Returns
    -------
    od : ndarray, shape (n_time, n_ch)
    """
    intensity = np.asarray(intensity, dtype=np.float64)

    if intensity.ndim == 1:
        intensity = intensity[:, np.newaxis]

    if baseline_end is None:
        baseline_end = intensity.shape[0]

    baseline = intensity[baseline_start:baseline_end, :]
    i0 = np.maximum(np.mean(baseline, axis=0, keepdims=True), _EPSILON)
    i_safe = np.maximum(intensity, _EPSILON)

    return -np.log10(i_safe / i0)
