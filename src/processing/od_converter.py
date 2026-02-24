"""
Optical Density (OD) Converter.

Converts raw intensity measurements to optical density:
    OD = -log10(I / I0)

where I0 is the mean intensity over a baseline period (or the
full recording if no baseline period is specified).
"""
from __future__ import annotations

import numpy as np


_EPSILON = 1e-10  # floor for intensity to avoid log(0)


def intensity_to_od(
    intensity: np.ndarray,
    baseline_start: int = 0,
    baseline_end: int | None = None,
) -> np.ndarray:
    """Convert raw intensity to optical density (OD).

    Parameters
    ----------
    intensity : ndarray, shape (n_time, n_ch)
        Raw optical intensity values.
    baseline_start : int
        Start index for the baseline averaging window.
    baseline_end : int or None
        End index (exclusive) for baseline. None = use entire recording.

    Returns
    -------
    od : ndarray, shape (n_time, n_ch)
        Optical density values.

    Notes
    -----
    - Negative or zero intensity values are clamped to ``_EPSILON``
      before computing the logarithm.
    - A per-channel baseline mean (I₀) is computed and clamped likewise.
    """
    intensity = np.asarray(intensity, dtype=np.float64)

    if intensity.ndim == 1:
        intensity = intensity[:, np.newaxis]

    # Determine baseline window
    if baseline_end is None:
        baseline_end = intensity.shape[0]

    baseline = intensity[baseline_start:baseline_end, :]

    # I0 = mean intensity over baseline, per channel
    i0 = np.mean(baseline, axis=0, keepdims=True)  # (1, n_ch)

    # Clamp to avoid log(0) or log(negative)
    i0 = np.maximum(i0, _EPSILON)
    i_safe = np.maximum(intensity, _EPSILON)

    # OD = -log10(I / I0)
    od = -np.log10(i_safe / i0)

    return od
