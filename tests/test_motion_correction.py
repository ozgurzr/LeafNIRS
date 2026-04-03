"""Tests for motion artifact detection and correction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from processing.motion_correction import (
    detect_artifacts,
    correct_tddr,
    correct_spline,
)
from processing.pipeline import ProcessingPipeline, PipelineState
from data_io.snirf_loader_base import ChannelInfo, ProbeGeometry


def _make_clean_od(n_time=1000, n_ch=4, fs=10.0):
    """Generate clean OD signal with slow hemodynamic fluctuation."""
    t = np.arange(n_time) / fs
    od = np.zeros((n_time, n_ch))
    for ch in range(n_ch):
        od[:, ch] = 0.01 * np.sin(2 * np.pi * 0.05 * t + ch * 0.5)
    return od, fs


def _inject_spikes(od, spike_indices, amplitude=0.5):
    """Inject motion artifact spikes into OD signal."""
    corrupted = od.copy()
    for idx in spike_indices:
        corrupted[idx, :] += amplitude
        if idx + 1 < corrupted.shape[0]:
            corrupted[idx + 1, :] -= amplitude * 0.5
    return corrupted




def test_detect_clean_signal():
    """Clean signal should have no artifacts detected."""
    od, fs = _make_clean_od()
    mask = detect_artifacts(od, fs)
    artifact_frac = mask.sum() / mask.size
    assert artifact_frac < 0.05, f"Too many artifacts in clean signal: {artifact_frac:.2%}"
    print(f"[PASS] Clean signal: {artifact_frac:.2%} flagged (< 5%)")


def test_detect_spike():
    """Large spike should be detected."""
    od, fs = _make_clean_od()
    spike_idx = [200, 500, 800]
    corrupted = _inject_spikes(od, spike_idx, amplitude=0.5)
    mask = detect_artifacts(corrupted, fs, threshold=3.0)
    for idx in spike_idx:
        window = mask[max(0, idx-2):idx+3, :]
        assert window.any(), f"Spike at index {idx} not detected"
    print(f"[PASS] Spikes at {spike_idx} detected")


def test_detect_returns_correct_shape():
    """Mask shape should match input."""
    od, fs = _make_clean_od(n_time=500, n_ch=8)
    mask = detect_artifacts(od, fs)
    assert mask.shape == (500, 8), f"Wrong shape: {mask.shape}"
    assert mask.dtype == bool
    print("[PASS] Detection mask shape and dtype correct")




def test_tddr_clean_signal():
    """TDDR on clean signal should not distort it significantly."""
    od, _ = _make_clean_od()
    corrected = correct_tddr(od)
    # Should be very similar to original
    rmse = np.sqrt(np.mean((corrected - od)**2))
    assert rmse < 0.005, f"TDDR distorted clean signal: RMSE={rmse:.4f}"
    print(f"[PASS] TDDR on clean signal: RMSE={rmse:.6f} (< 0.005)")


def test_tddr_removes_spikes():
    """TDDR should reduce spike amplitude."""
    od, _ = _make_clean_od()
    spike_idx = [200, 500, 800]
    corrupted = _inject_spikes(od, spike_idx, amplitude=0.5)

    corrected = correct_tddr(corrupted)

    # Check that spike regions are reduced
    for idx in spike_idx:
        original_spike = np.abs(corrupted[idx, 0] - od[idx, 0])
        corrected_spike = np.abs(corrected[idx, 0] - od[idx, 0])
        assert corrected_spike < original_spike, (
            f"Spike at {idx}: original={original_spike:.4f}, "
            f"corrected={corrected_spike:.4f}"
        )
    print("[PASS] TDDR reduces spike amplitudes")


def test_tddr_output_shape():
    """Output shape should match input."""
    od, _ = _make_clean_od(n_time=300, n_ch=6)
    corrected = correct_tddr(od)
    assert corrected.shape == od.shape
    assert np.all(np.isfinite(corrected))
    print("[PASS] TDDR output shape and finiteness correct")




def test_spline_removes_artifacts():
    """Spline should replace artifact segments with interpolated values."""
    od, fs = _make_clean_od()
    spike_idx = [200, 201, 202, 500, 501, 502]
    corrupted = _inject_spikes(od, spike_idx, amplitude=0.5)
    mask = np.zeros_like(od, dtype=bool)
    for idx in spike_idx:
        mask[idx, :] = True

    corrected = correct_spline(corrupted, mask)

    # Artifact regions should now be closer to clean signal
    for idx in [200, 500]:
        orig_err = np.abs(corrupted[idx, 0] - od[idx, 0])
        corr_err = np.abs(corrected[idx, 0] - od[idx, 0])
        assert corr_err < orig_err, (
            f"Spline at {idx}: orig_err={orig_err:.4f}, corr_err={corr_err:.4f}"
        )
    print("[PASS] Spline correction reduces artifact amplitude")


def test_spline_no_artifacts():
    """Spline with empty mask should return original signal."""
    od, _ = _make_clean_od()
    mask = np.zeros_like(od, dtype=bool)
    corrected = correct_spline(od, mask)
    assert np.allclose(corrected, od), "Spline with no artifacts should return original"
    print("[PASS] Spline with no artifacts returns original")




def _make_test_pipeline():
    """Create a pipeline with synthetic data for testing."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)
    n_time = len(t)
    n_ch = 4

    channels = []
    for pair in range(2):
        for wl in [1, 2]:
            channels.append(ChannelInfo(
                source_index=pair + 1,
                detector_index=1,
                wavelength_index=wl,
                data_type=1,
            ))

    probe = ProbeGeometry(
        source_pos=np.array([[0, 0], [30, 0]]),
        detector_pos=np.array([[15, 0]]),
        wavelengths=np.array([760.0, 850.0]),
    )

    intensity = np.ones((n_time, n_ch)) * 1000.0
    for i in range(n_ch):
        intensity[:, i] += 20 * np.sin(2 * np.pi * 0.05 * t)

    return ProcessingPipeline(intensity, fs, channels=channels, probe=probe)


def test_pipeline_corrected_state():
    """Pipeline should support CORRECTED state."""
    pipe = _make_test_pipeline()
    pipe.convert_to_od()
    pipe.apply_motion_correction(method='tddr')
    assert pipe.state == PipelineState.CORRECTED
    assert pipe.result.corrected is not None
    print(f"[PASS] Pipeline CORRECTED state: shape={pipe.result.corrected.shape}")


def test_pipeline_full_with_correction():
    """Full pipeline: intensity → OD → correct → filter → MBLL."""
    pipe = _make_test_pipeline()
    pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    # Bandpass should auto-run OD + correction if in auto chain
    assert pipe.result.od is not None
    pipe.convert_to_concentration()
    assert pipe.state == PipelineState.CONCENTRATION
    assert pipe.result.hbo is not None
    print("[PASS] Full pipeline with motion correction works")


def test_pipeline_view_switching_with_corrected():
    """Can switch to CORRECTED view."""
    pipe = _make_test_pipeline()
    pipe.convert_to_od()
    pipe.apply_motion_correction(method='tddr')
    pipe.set_view(PipelineState.OD)
    assert pipe.state == PipelineState.OD
    pipe.set_view(PipelineState.CORRECTED)
    assert pipe.state == PipelineState.CORRECTED
    print("[PASS] View switching with CORRECTED state works")




if __name__ == '__main__':
    tests = [
        test_detect_clean_signal,
        test_detect_spike,
        test_detect_returns_correct_shape,
        test_tddr_clean_signal,
        test_tddr_removes_spikes,
        test_tddr_output_shape,
        test_spline_removes_artifacts,
        test_spline_no_artifacts,
        test_pipeline_corrected_state,
        test_pipeline_full_with_correction,
        test_pipeline_view_switching_with_corrected,
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
        print("[OK] All motion correction tests passed!")
    else:
        print("[!!] Some tests failed")
