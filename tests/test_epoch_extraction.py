"""Tests for epoch extraction and block averaging."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from processing.epoch_extraction import (
    extract_epochs,
    baseline_correct,
    block_average,
    compute_condition_average,
    BlockAverageResult,
)


def _make_synthetic_hrf(n_time=3000, fs=10.0, n_pairs=4, n_stim=5):
    """Create synthetic HbO/HbR with known HRF responses at stimulus onsets."""
    t = np.arange(n_time) / fs
    hbo = np.random.randn(n_time, n_pairs) * 0.01  # small noise
    hbr = np.random.randn(n_time, n_pairs) * 0.01

    # Stimulus onsets every ~50s starting at 10s
    onsets = np.array([10.0 + i * 50.0 for i in range(n_stim)])
    # Filter onsets that would extend past recording
    max_onset = t[-1] - 25.0  # need 20s post + margin
    onsets = onsets[onsets <= max_onset]

    # Inject canonical HRF-like response at each onset
    for onset in onsets:
        idx = int(onset * fs)
        # Simple gamma-like HRF: peaks at ~5s
        for dt in range(int(20 * fs)):
            if idx + dt < n_time:
                t_rel = dt / fs
                hrf_val = t_rel * np.exp(-t_rel / 1.5) * 0.5  # peaks around 1.5s
                hbo[idx + dt, :] += hrf_val
                hbr[idx + dt, :] -= hrf_val * 0.3  # HbR is inverted, smaller

    return hbo, hbr, t, onsets




def test_extract_epoch_shape():
    """Extracted epochs should have correct shape."""
    hbo, _, t, onsets = _make_synthetic_hrf()
    epochs, epoch_time = extract_epochs(hbo, t, onsets, pre_sec=2.0, post_sec=20.0)

    fs = 10.0
    expected_time_len = int(2.0 * fs) + int(20.0 * fs)  # 220 samples
    assert epochs.shape[1] == expected_time_len, f"Wrong epoch length: {epochs.shape[1]}"
    assert epochs.shape[2] == hbo.shape[1], f"Wrong n_signals: {epochs.shape[2]}"
    assert len(epoch_time) == expected_time_len
    assert epochs.shape[0] <= len(onsets), "More epochs than onsets"
    print(f"[PASS] Epoch shape: {epochs.shape}, time: {epoch_time.shape}")


def test_extract_epoch_time_axis():
    """Epoch time axis should span [-pre, +post)."""
    hbo, _, t, onsets = _make_synthetic_hrf()
    _, epoch_time = extract_epochs(hbo, t, onsets, pre_sec=2.0, post_sec=20.0)

    assert epoch_time[0] < 0, f"First time should be negative: {epoch_time[0]}"
    assert epoch_time[-1] > 0, f"Last time should be positive: {epoch_time[-1]}"
    assert abs(epoch_time[0] - (-2.0)) < 0.15, f"Pre-stim start wrong: {epoch_time[0]}"
    print(f"[PASS] Epoch time: [{epoch_time[0]:.2f}s, {epoch_time[-1]:.2f}s]")


def test_extract_skips_boundary_onsets():
    """Onsets too close to start/end should be skipped."""
    n_time, fs = 500, 10.0
    t = np.arange(n_time) / fs  # 50s recording
    data = np.random.randn(n_time, 2)

    # Onset at 1s (not enough pre-stimulus) and 49s (not enough post-stimulus)
    onsets = np.array([1.0, 25.0, 49.0])
    epochs, _ = extract_epochs(data, t, onsets, pre_sec=5.0, post_sec=20.0)

    assert epochs.shape[0] == 1, f"Should skip 2 boundary onsets, got {epochs.shape[0]} trials"
    print(f"[PASS] Boundary onsets skipped: {epochs.shape[0]} valid of {len(onsets)}")


def test_extract_empty_onsets():
    """Empty onsets should return empty epochs."""
    data = np.random.randn(1000, 3)
    t = np.arange(1000) / 10.0
    epochs, epoch_time = extract_epochs(data, t, np.array([]), pre_sec=2.0, post_sec=20.0)

    assert epochs.shape[0] == 0
    print("[PASS] Empty onsets → empty epochs")




def test_baseline_correct():
    """Baseline correction should zero the pre-stimulus mean."""
    n_trials, n_time, n_sig = 5, 220, 3
    fs = 10.0
    n_pre = 20  # 2s * 10Hz

    # Create epochs with known baseline offset
    epoch_time = np.arange(-n_pre, n_time - n_pre) / fs
    epochs = np.ones((n_trials, n_time, n_sig)) * 5.0  # baseline of 5.0

    corrected = baseline_correct(epochs, epoch_time)

    # Pre-stimulus mean should now be ~0
    baseline_mask = epoch_time < 0
    baseline_mean = corrected[:, baseline_mask, :].mean()
    assert abs(baseline_mean) < 1e-10, f"Baseline mean should be ~0: {baseline_mean}"
    print(f"[PASS] Baseline corrected: mean={baseline_mean:.2e}")


def test_baseline_preserves_response():
    """Baseline correction should preserve the post-stimulus shape."""
    n_trials, n_time, n_sig = 3, 220, 2
    fs = 10.0
    n_pre = 20

    epoch_time = np.arange(-n_pre, n_time - n_pre) / fs
    epochs = np.zeros((n_trials, n_time, n_sig))

    # Add baseline offset + post-stimulus response
    epochs[:, :, :] = 2.0  # baseline
    epochs[:, n_pre + 50:n_pre + 60, :] = 7.0  # response peak

    corrected = baseline_correct(epochs, epoch_time)

    # Response should be ~5.0 (7.0 - 2.0 baseline)
    peak = corrected[:, n_pre + 55, 0].mean()
    assert abs(peak - 5.0) < 0.1, f"Peak should be ~5.0: {peak}"
    print(f"[PASS] Response preserved after baseline: peak={peak:.2f}")




def test_block_average_mean():
    """Block average mean should match expected value."""
    n_trials, n_time, n_sig = 10, 100, 2
    # All trials identical
    trial = np.sin(np.linspace(0, 2*np.pi, n_time))[:, np.newaxis] * np.ones((1, n_sig))
    epochs = np.stack([trial] * n_trials)

    mean, sem = block_average(epochs)
    assert mean.shape == (n_time, n_sig)
    assert np.allclose(mean, trial, atol=1e-10), "Mean should match trial"
    assert np.allclose(sem, 0, atol=1e-10), "SEM should be 0 for identical trials"
    print("[PASS] Block average of identical trials: mean correct, SEM=0")


def test_block_average_sem():
    """SEM should decrease with more trials."""
    n_time, n_sig = 100, 1
    np.random.seed(42)

    # Few trials
    epochs_few = np.random.randn(5, n_time, n_sig)
    _, sem_few = block_average(epochs_few)

    # Many trials
    epochs_many = np.random.randn(50, n_time, n_sig)
    _, sem_many = block_average(epochs_many)

    assert np.mean(sem_many) < np.mean(sem_few), "SEM should decrease with more trials"
    print(f"[PASS] SEM decreases: few={np.mean(sem_few):.4f} > many={np.mean(sem_many):.4f}")


def test_block_average_empty():
    """Empty epochs should return zeros."""
    mean, sem = block_average(np.empty((0, 100, 3)))
    assert mean.shape == (100, 3)
    assert np.all(mean == 0)
    print("[PASS] Empty epochs → zero mean/sem")




def test_compute_condition_average():
    """Full pipeline: extract → baseline → average."""
    hbo, hbr, t, onsets = _make_synthetic_hrf(n_stim=5)
    pair_labels = [f"S1-D{i+1}" for i in range(hbo.shape[1])]

    result = compute_condition_average(
        hbo, hbr, t, onsets, "task", pair_labels,
        pre_sec=2.0, post_sec=20.0,
    )

    assert isinstance(result, BlockAverageResult)
    assert result.condition == "task"
    assert result.n_trials > 0
    assert result.hbo_mean.shape[0] == len(result.epoch_time)
    assert result.hbo_mean.shape[1] == len(pair_labels)

    # HbO peak should be positive (we injected positive HRF)
    peak_idx = np.argmax(result.hbo_mean[:, 0])
    assert result.hbo_mean[peak_idx, 0] > 0, "HbO peak should be positive"

    # HbR should be inverted (negative)
    hbr_at_peak = result.hbr_mean[peak_idx, 0]
    assert hbr_at_peak < 0, f"HbR should be negative at HbO peak: {hbr_at_peak}"

    print(
        f"[PASS] Condition average: {result.n_trials} trials, "
        f"HbO peak={result.hbo_mean[peak_idx, 0]:.4f} @ {result.epoch_time[peak_idx]:.1f}s, "
        f"HbR={hbr_at_peak:.4f}"
    )




if __name__ == '__main__':
    tests = [
        test_extract_epoch_shape,
        test_extract_epoch_time_axis,
        test_extract_skips_boundary_onsets,
        test_extract_empty_onsets,
        test_baseline_correct,
        test_baseline_preserves_response,
        test_block_average_mean,
        test_block_average_sem,
        test_block_average_empty,
        test_compute_condition_average,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("[OK] All epoch extraction tests passed!")
    else:
        print("[!!] Some tests failed")
