"""
Tests for Phase 2 — Signal Processing.

Covers:
- OD conversion correctness (known values)
- Bandpass filter frequency response
- Pipeline state management
- Edge cases
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from processing.od_converter import intensity_to_od
from processing.bandpass_filter import bandpass_filter
from processing.pipeline import ProcessingPipeline, PipelineState


# ══════════════════════════════════════════
#  OD Conversion Tests
# ══════════════════════════════════════════

def test_od_known_values():
    """OD of constant signal should be zero."""
    intensity = np.ones((100, 4)) * 500.0
    od = intensity_to_od(intensity)
    assert od.shape == intensity.shape
    assert np.allclose(od, 0.0, atol=1e-10), f"Expected all zeros, got max={od.max()}"
    print("[PASS] OD of constant signal = 0")


def test_od_half_intensity():
    """Halving intensity from baseline should give OD = log10(2) ≈ 0.301."""
    # Baseline: first 50 samples at 1000, then drop to 500
    intensity = np.ones((100, 1)) * 1000.0
    intensity[50:, :] = 500.0
    od = intensity_to_od(intensity, baseline_start=0, baseline_end=50)
    expected_od = np.log10(2)  # ≈ 0.301
    actual = od[75, 0]
    assert abs(actual - expected_od) < 0.001, f"Expected ~{expected_od:.4f}, got {actual:.4f}"
    print(f"[PASS] Half intensity OD = {actual:.4f} (expected {expected_od:.4f})")


def test_od_zero_intensity():
    """Zero intensity should be clamped, not produce -inf."""
    intensity = np.array([[1000], [0], [500]], dtype=np.float64)
    od = intensity_to_od(intensity)
    assert np.all(np.isfinite(od)), "OD contains non-finite values for zero input!"
    print("[PASS] Zero intensity handled without inf/nan")


def test_od_negative_intensity():
    """Negative intensity should be clamped to epsilon."""
    intensity = np.array([[1000], [-100], [500]], dtype=np.float64)
    od = intensity_to_od(intensity)
    assert np.all(np.isfinite(od)), "OD contains non-finite values for negative input!"
    print("[PASS] Negative intensity handled without inf/nan")


def test_od_multichannel():
    """OD conversion should work independently per channel."""
    intensity = np.column_stack([
        np.ones(100) * 1000,
        np.ones(100) * 2000,
        np.ones(100) * 500,
    ])
    od = intensity_to_od(intensity)
    assert od.shape == (100, 3)
    assert np.allclose(od, 0.0, atol=1e-10)
    print("[PASS] Multi-channel OD works independently")


# ══════════════════════════════════════════
#  Bandpass Filter Tests
# ══════════════════════════════════════════

def test_filter_removes_dc():
    """DC component (0 Hz) should be removed by the highpass."""
    fs = 10.0
    t = np.arange(0, 60, 1/fs)  # 60 seconds
    signal = np.ones((len(t), 1)) * 100.0  # pure DC
    filtered = bandpass_filter(signal, fs, low=0.01, high=0.1)
    assert np.max(np.abs(filtered)) < 1.0, f"DC not removed, max={np.max(np.abs(filtered)):.3f}"
    print("[PASS] DC component removed by highpass")


def test_filter_passes_target_band():
    """A 0.05 Hz signal (within passband) should be mostly preserved."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)  # Need longer signal for low freqs
    f_signal = 0.05  # Hz, within 0.01-0.1 band
    signal = np.sin(2 * np.pi * f_signal * t).reshape(-1, 1)
    filtered = bandpass_filter(signal, fs, low=0.01, high=0.1)
    # After filtering, amplitude should be close to 1.0 (within -3dB = 0.707)
    # Skip edges (transient effects)
    steady = filtered[len(t)//4:-len(t)//4, 0]
    amp = np.max(np.abs(steady))
    assert amp > 0.7, f"Passband signal attenuated too much: amp={amp:.3f}"
    print(f"[PASS] 0.05 Hz signal preserved (amplitude={amp:.3f})")


def test_filter_removes_cardiac():
    """A 1 Hz signal (cardiac) should be strongly attenuated."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)  # 120s for stable transients
    signal = np.sin(2 * np.pi * 1.0 * t).reshape(-1, 1)  # 1 Hz cardiac
    filtered = bandpass_filter(signal, fs, low=0.01, high=0.1)
    # Skip transient edges, measure steady-state attenuation
    mid = filtered[len(t)//4:-len(t)//4, 0]
    ratio = np.max(np.abs(mid)) / np.max(np.abs(signal))
    assert ratio < 0.05, f"Cardiac not sufficiently removed: ratio={ratio:.4f}"
    print(f"[PASS] 1 Hz cardiac removed (attenuation ratio={ratio:.4f})")


def test_filter_invalid_cutoffs():
    """Invalid cutoff frequencies should raise ValueError."""
    signal = np.random.randn(1000, 1)
    errors = 0

    try:
        bandpass_filter(signal, 10.0, low=0.5, high=0.1)
    except ValueError:
        errors += 1

    try:
        bandpass_filter(signal, 10.0, low=0.01, high=6.0)  # > Nyquist
    except ValueError:
        errors += 1

    try:
        bandpass_filter(signal, 10.0, low=-0.01, high=0.1)
    except ValueError:
        errors += 1

    assert errors == 3, f"Expected 3 ValueError, got {errors}"
    print("[PASS] Invalid cutoffs raise ValueError")


def test_filter_short_signal():
    """Signal shorter than minimum length should raise ValueError."""
    signal = np.random.randn(10, 1)  # Too short for order-3 filter
    try:
        bandpass_filter(signal, 10.0, low=0.01, high=0.1, order=3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] Short signal raises ValueError")


# ══════════════════════════════════════════
#  Pipeline Tests
# ══════════════════════════════════════════

def test_pipeline_initial_state():
    """Pipeline starts in RAW state."""
    intensity = np.random.rand(500, 4) * 1000 + 100
    pipe = ProcessingPipeline(intensity, sampling_rate=10.0)
    assert pipe.state == PipelineState.RAW
    assert np.array_equal(pipe.result.active_data, intensity)
    print("[PASS] Pipeline starts in RAW state")


def test_pipeline_od_conversion():
    """Converting to OD advances state and produces valid output."""
    intensity = np.random.rand(500, 4) * 1000 + 100
    pipe = ProcessingPipeline(intensity, sampling_rate=10.0)
    od = pipe.convert_to_od()
    assert pipe.state == PipelineState.OD
    assert od.shape == intensity.shape
    assert np.all(np.isfinite(od))
    print("[PASS] Pipeline OD conversion works")


def test_pipeline_bandpass():
    """Applying bandpass advances state to FILTERED."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)
    intensity = np.column_stack([
        1000 + 50 * np.sin(2 * np.pi * 0.05 * t),
        800 + 30 * np.sin(2 * np.pi * 0.05 * t),
    ])
    pipe = ProcessingPipeline(intensity, sampling_rate=fs)
    filtered = pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    assert pipe.state == PipelineState.FILTERED
    assert filtered.shape == intensity.shape
    assert np.all(np.isfinite(filtered))
    print("[PASS] Pipeline bandpass works")


def test_pipeline_view_switching():
    """Can switch between RAW, OD, and FILTERED views."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)
    intensity = np.ones((len(t), 2)) * 1000
    pipe = ProcessingPipeline(intensity, sampling_rate=fs)
    pipe.apply_bandpass(low=0.01, high=0.1)

    # Switch to each view
    raw = pipe.set_view(PipelineState.RAW)
    assert pipe.state == PipelineState.RAW
    assert np.array_equal(raw, intensity)

    od = pipe.set_view(PipelineState.OD)
    assert pipe.state == PipelineState.OD

    filt = pipe.set_view(PipelineState.FILTERED)
    assert pipe.state == PipelineState.FILTERED
    print("[PASS] Pipeline view switching works")


def test_pipeline_reset():
    """Reset clears all processed data."""
    intensity = np.random.rand(500, 2) * 1000 + 100
    pipe = ProcessingPipeline(intensity, sampling_rate=10.0)
    pipe.apply_bandpass(low=0.01, high=0.1)
    pipe.reset()
    assert pipe.state == PipelineState.RAW
    assert pipe.result.od is None
    assert pipe.result.filtered is None
    print("[PASS] Pipeline reset works")


# ══════════════════════════════════════════
#  Run all tests
# ══════════════════════════════════════════

if __name__ == '__main__':
    tests = [
        test_od_known_values,
        test_od_half_intensity,
        test_od_zero_intensity,
        test_od_negative_intensity,
        test_od_multichannel,
        test_filter_removes_dc,
        test_filter_passes_target_band,
        test_filter_removes_cardiac,
        test_filter_invalid_cutoffs,
        test_filter_short_signal,
        test_pipeline_initial_state,
        test_pipeline_od_conversion,
        test_pipeline_bandpass,
        test_pipeline_view_switching,
        test_pipeline_reset,
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
        print("[OK] All processing tests passed!")
    else:
        print("[!!] Some tests failed")
