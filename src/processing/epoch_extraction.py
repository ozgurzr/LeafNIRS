"""Epoch extraction and block averaging for task-based fNIRS."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BlockAverageResult:
    """Block-averaged HRF for one stimulus condition."""
    condition: str
    epoch_time: np.ndarray
    hbo_mean: np.ndarray
    hbo_sem: np.ndarray
    hbr_mean: np.ndarray
    hbr_sem: np.ndarray
    n_trials: int
    pair_labels: list[str] = field(default_factory=list)


def extract_epochs(
    data: np.ndarray,
    time: np.ndarray,
    onsets: np.ndarray,
    pre_sec: float = 2.0,
    post_sec: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time-locked epochs around stimulus onsets.

    Parameters
    ----------
    data : ndarray, shape (n_time, n_signals)
    time : ndarray, shape (n_time,)
    onsets : ndarray — stimulus onset times in seconds
    pre_sec, post_sec : float — epoch window

    Returns
    -------
    epochs : ndarray, shape (n_valid_trials, n_epoch_time, n_signals)
    epoch_time : ndarray, shape (n_epoch_time,)
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    fs = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 1.0
    n_pre = int(round(pre_sec * fs))
    n_post = int(round(post_sec * fs))
    n_epoch = n_pre + n_post
    n_time, n_signals = data.shape

    epoch_time = np.arange(-n_pre, n_post) / fs

    epochs = []
    for onset in onsets:
        idx = np.argmin(np.abs(time - onset))
        start = idx - n_pre
        end = idx + n_post

        if start < 0 or end > n_time:
            continue
        epochs.append(data[start:end, :])

    if len(epochs) == 0:
        return np.empty((0, n_epoch, n_signals)), epoch_time
    return np.array(epochs), epoch_time


def baseline_correct(
    epochs: np.ndarray,
    epoch_time: np.ndarray,
) -> np.ndarray:
    """Subtract pre-stimulus (epoch_time < 0) baseline mean from each epoch."""
    if epochs.shape[0] == 0:
        return epochs

    baseline_mask = epoch_time < 0
    if not np.any(baseline_mask):
        return epochs

    baseline_mean = epochs[:, baseline_mask, :].mean(axis=1, keepdims=True)
    return epochs - baseline_mean


def block_average(
    epochs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and SEM across trials.

    Returns
    -------
    mean, sem : ndarray, shape (n_epoch_time, n_signals)
    """
    if epochs.shape[0] == 0:
        n_time = epochs.shape[1] if epochs.ndim > 1 else 0
        n_sig = epochs.shape[2] if epochs.ndim > 2 else 0
        return np.zeros((n_time, n_sig)), np.zeros((n_time, n_sig))

    n_trials = epochs.shape[0]
    mean = np.mean(epochs, axis=0)
    std = np.std(epochs, axis=0, ddof=1) if n_trials > 1 else np.zeros_like(mean)
    return mean, std / np.sqrt(n_trials)


def compute_condition_average(
    hbo: np.ndarray,
    hbr: np.ndarray,
    time: np.ndarray,
    onsets: np.ndarray,
    condition_name: str,
    pair_labels: list[str],
    pre_sec: float = 2.0,
    post_sec: float = 20.0,
) -> BlockAverageResult:
    """Full pipeline: extract → baseline correct → block average."""
    hbo_epochs, epoch_time = extract_epochs(hbo, time, onsets, pre_sec, post_sec)
    hbr_epochs, _ = extract_epochs(hbr, time, onsets, pre_sec, post_sec)

    hbo_epochs = baseline_correct(hbo_epochs, epoch_time)
    hbr_epochs = baseline_correct(hbr_epochs, epoch_time)

    hbo_mean, hbo_sem = block_average(hbo_epochs)
    hbr_mean, hbr_sem = block_average(hbr_epochs)

    return BlockAverageResult(
        condition=condition_name,
        epoch_time=epoch_time,
        hbo_mean=hbo_mean,
        hbo_sem=hbo_sem,
        hbr_mean=hbr_mean,
        hbr_sem=hbr_sem,
        n_trials=hbo_epochs.shape[0],
        pair_labels=list(pair_labels),
    )
