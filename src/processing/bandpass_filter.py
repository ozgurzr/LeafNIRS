"""Butterworth bandpass filter for fNIRS signals (zero-phase via filtfilt)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float = 0.01,
    high: float = 0.1,
    order: int = 3,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : ndarray, shape (n_time, n_ch)
    fs : float
        Sampling frequency in Hz.
    low, high : float
        Cutoff frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered : ndarray, same shape as data
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    nyquist = fs / 2.0

    if low <= 0:
        raise ValueError(f"Low cutoff must be > 0 Hz, got {low}")
    if high >= nyquist:
        raise ValueError(f"High cutoff ({high} Hz) must be < Nyquist ({nyquist} Hz)")
    if low >= high:
        raise ValueError(f"Low cutoff ({low} Hz) must be < high cutoff ({high} Hz)")

    min_len = 3 * (2 * order + 1)
    if data.shape[0] < min_len:
        raise ValueError(
            f"Signal too short ({data.shape[0]} samples). "
            f"Need at least {min_len} for order-{order} filter."
        )

    wn = [low / nyquist, high / nyquist]
    b, a = butter(order, wn, btype='bandpass')

    filtered = np.empty_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered
