"""
Bandpass Filter for fNIRS signals.

Applies a zero-phase Butterworth bandpass filter to isolate the
hemodynamic response frequency band (typically 0.01–0.1 Hz),
removing baseline drift and physiological noise.
"""
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
        Input signal (typically optical density).
    fs : float
        Sampling frequency in Hz.
    low : float
        Low cutoff frequency in Hz (removes slow drift).
    high : float
        High cutoff frequency in Hz (removes cardiac/respiratory).
    order : int
        Filter order (default 3).

    Returns
    -------
    filtered : ndarray, same shape as ``data``
        Bandpass-filtered signal.

    Raises
    ------
    ValueError
        If cutoff frequencies are invalid or signal is too short.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    nyquist = fs / 2.0

    # Validate
    if low <= 0:
        raise ValueError(f"Low cutoff must be > 0 Hz, got {low}")
    if high >= nyquist:
        raise ValueError(
            f"High cutoff ({high} Hz) must be < Nyquist ({nyquist} Hz)"
        )
    if low >= high:
        raise ValueError(
            f"Low cutoff ({low} Hz) must be < high cutoff ({high} Hz)"
        )

    # Minimum signal length for filtfilt: 3 * max(len(a), len(b))
    # For order N butterworth: filter length = 2*N+1
    min_len = 3 * (2 * order + 1)
    if data.shape[0] < min_len:
        raise ValueError(
            f"Signal too short ({data.shape[0]} samples). "
            f"Need at least {min_len} for order-{order} filter."
        )

    # Design Butterworth bandpass filter
    wn = [low / nyquist, high / nyquist]
    b, a = butter(order, wn, btype='bandpass')

    # Apply zero-phase filtering to each channel
    filtered = np.empty_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered
