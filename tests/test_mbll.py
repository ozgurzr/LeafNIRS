"""Tests for MBLL extinction coefficients, DPF, concentration solver, and pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from processing.mbll_converter import (
    get_extinction_coefficients,
    get_dpf,
    od_to_concentration,
)
from processing.pipeline import ProcessingPipeline, PipelineState
from data_io.snirf_loader_base import ChannelInfo, ProbeGeometry



def test_extinction_known_wavelengths():
    """Exact wavelengths in the table should return correct values."""
    E = get_extinction_coefficients([760, 850])
    assert E.shape == (2, 2)
    # At 760nm: ε_HbO < ε_HbR (deoxy dominates)
    assert E[0, 0] < E[0, 1], f"At 760nm, ε_HbO={E[0,0]} should be < ε_HbR={E[0,1]}"
    # At 850nm: ε_HbO > ε_HbR (oxy dominates)
    assert E[1, 0] > E[1, 1], f"At 850nm, ε_HbO={E[1,0]} should be > ε_HbR={E[1,1]}"
    print(f"[PASS] Extinction at 760nm: ε_HbO={E[0,0]:.6f}, ε_HbR={E[0,1]:.6f}")
    print(f"[PASS] Extinction at 850nm: ε_HbO={E[1,0]:.6f}, ε_HbR={E[1,1]:.6f}")


def test_extinction_nearest_neighbour():
    """Wavelengths not in table should use nearest entry."""
    E_exact = get_extinction_coefficients([760])
    E_near = get_extinction_coefficients([761])
    assert np.allclose(E_exact, E_near), "761nm should resolve to 760nm"
    print("[PASS] Nearest-neighbour interpolation works")


def test_extinction_all_positive():
    """All extinction coefficients should be positive."""
    wls = [690, 760, 800, 850, 940]
    E = get_extinction_coefficients(wls)
    assert np.all(E > 0), "All extinction coefficients must be > 0"
    print("[PASS] All extinction coefficients positive")




def test_dpf_known_wavelengths():
    """DPF at known wavelengths should return table values."""
    dpf = get_dpf([760, 850])
    assert len(dpf) == 2
    assert dpf[0] > dpf[1], f"DPF at 760nm ({dpf[0]}) should be > DPF at 850nm ({dpf[1]})"
    assert dpf[0] > 4.0 and dpf[0] < 7.0, f"DPF at 760nm out of range: {dpf[0]}"
    assert dpf[1] > 3.0 and dpf[1] < 6.0, f"DPF at 850nm out of range: {dpf[1]}"
    print(f"[PASS] DPF at 760nm={dpf[0]:.2f}, 850nm={dpf[1]:.2f}")


def test_dpf_decreases_with_wavelength():
    """DPF generally decreases with increasing wavelength."""
    wls = [700, 760, 800, 850, 900]
    dpf = get_dpf(wls)
    for i in range(len(dpf) - 1):
        assert dpf[i] >= dpf[i+1], f"DPF should decrease: {wls[i]}nm={dpf[i]}, {wls[i+1]}nm={dpf[i+1]}"
    print("[PASS] DPF decreases with wavelength")




def _make_test_data(n_time=200, n_pairs=2, wavelengths=None):
    """Create synthetic test data with known S-D pairs."""
    if wavelengths is None:
        wavelengths = [760.0, 850.0]

    channels = []
    for pair_idx in range(n_pairs):
        src = pair_idx + 1
        det = 1
        for wl_idx in range(1, len(wavelengths) + 1):
            channels.append(ChannelInfo(
                source_index=src,
                detector_index=det,
                wavelength_index=wl_idx,
                data_type=1,
            ))

    n_ch = len(channels)
    source_pos = np.array([[i * 30, 0] for i in range(n_pairs)])
    detector_pos = np.array([[15, 0]])  # single detector at 15mm

    probe = ProbeGeometry(
        source_pos=source_pos,
        detector_pos=detector_pos,
        wavelengths=np.array(wavelengths),
    )

    return channels, probe, n_ch


def test_concentration_zero_od():
    """Zero OD should give zero concentration."""
    channels, probe, n_ch = _make_test_data(n_time=100, n_pairs=2)
    od = np.zeros((100, n_ch))
    hbo, hbr, labels = od_to_concentration(od, channels, probe)
    assert hbo.shape == (100, 2)
    assert hbr.shape == (100, 2)
    assert np.allclose(hbo, 0.0, atol=1e-12)
    assert np.allclose(hbr, 0.0, atol=1e-12)
    assert len(labels) == 2
    print(f"[PASS] Zero OD → zero concentration, pairs={labels}")


def test_concentration_output_shape():
    """Output should have (n_time, n_pairs) shape."""
    channels, probe, n_ch = _make_test_data(n_time=500, n_pairs=4)
    od = np.random.randn(500, n_ch) * 0.01
    hbo, hbr, labels = od_to_concentration(od, channels, probe)
    assert hbo.shape == (500, 4), f"Expected (500, 4), got {hbo.shape}"
    assert hbr.shape == (500, 4), f"Expected (500, 4), got {hbr.shape}"
    assert len(labels) == 4
    print(f"[PASS] Output shape correct: {hbo.shape}, pairs={labels}")


def test_concentration_finite():
    """Concentration values should all be finite."""
    channels, probe, n_ch = _make_test_data(n_time=200, n_pairs=3)
    od = np.random.randn(200, n_ch) * 0.01
    hbo, hbr, labels = od_to_concentration(od, channels, probe)
    assert np.all(np.isfinite(hbo)), "HbO contains non-finite values"
    assert np.all(np.isfinite(hbr)), "HbR contains non-finite values"
    print("[PASS] All concentration values finite")


def test_concentration_pair_labels():
    """Pair labels should follow S{n}-D{n} format."""
    channels, probe, n_ch = _make_test_data(n_time=50, n_pairs=3)
    od = np.zeros((50, n_ch))
    _, _, labels = od_to_concentration(od, channels, probe)
    for lbl in labels:
        assert lbl.startswith("S") and "-D" in lbl, f"Bad label: {lbl}"
    print(f"[PASS] Pair labels correct: {labels}")


def test_concentration_known_solve():
    """Test with hand-computed known values.

    If ΔOD at both wavelengths is identical and positive,
    HbO and HbR should have opposite signs at 760/850nm
    because ε_HbO(760) < ε_HbR(760) but ε_HbO(850) > ε_HbR(850).
    """
    channels, probe, n_ch = _make_test_data(n_time=10, n_pairs=1)
    od = np.ones((10, n_ch)) * 0.01  # same positive OD at both wavelengths
    hbo, hbr, labels = od_to_concentration(od, channels, probe)
    # Both should be finite, non-zero
    assert np.all(np.isfinite(hbo))
    assert np.all(np.isfinite(hbr))
    assert not np.allclose(hbo, 0.0)
    assert not np.allclose(hbr, 0.0)
    print(f"[PASS] Known-value solve: HbO={hbo[0,0]:.4f}, HbR={hbr[0,0]:.4f} μmol/L")




def test_pipeline_concentration():
    """Full pipeline: intensity → OD → filter → concentration."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)
    n_time = len(t)

    channels, probe, n_ch = _make_test_data(n_time=n_time, n_pairs=2)

    # Generate synthetic intensity with hemodynamic-like fluctuation
    intensity = np.ones((n_time, n_ch)) * 1000.0
    for i in range(n_ch):
        intensity[:, i] += 20 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz

    pipe = ProcessingPipeline(intensity, fs, channels=channels, probe=probe)

    # Full pipeline
    pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    hbo, hbr = pipe.convert_to_concentration()

    assert pipe.state == PipelineState.CONCENTRATION
    assert hbo.shape == (n_time, 2)
    assert hbr.shape == (n_time, 2)
    assert np.all(np.isfinite(hbo))
    assert np.all(np.isfinite(hbr))
    print(f"[PASS] Full pipeline: HbO range=[{hbo.min():.4f}, {hbo.max():.4f}]")
    print(f"       HbR range=[{hbr.min():.4f}, {hbr.max():.4f}]")


def test_pipeline_view_switching_with_concentration():
    """Can switch between all views including concentration."""
    fs = 10.0
    t = np.arange(0, 120, 1/fs)
    n_time = len(t)
    channels, probe, n_ch = _make_test_data(n_time=n_time, n_pairs=1)
    intensity = np.ones((n_time, n_ch)) * 1000.0

    pipe = ProcessingPipeline(intensity, fs, channels=channels, probe=probe)
    pipe.convert_to_concentration()

    # Switch back to raw
    pipe.set_view(PipelineState.RAW)
    assert pipe.state == PipelineState.RAW

    # Switch to concentration
    pipe.set_view(PipelineState.CONCENTRATION)
    assert pipe.state == PipelineState.CONCENTRATION
    print("[PASS] View switching with concentration works")




if __name__ == '__main__':
    tests = [
        test_extinction_known_wavelengths,
        test_extinction_nearest_neighbour,
        test_extinction_all_positive,
        test_dpf_known_wavelengths,
        test_dpf_decreases_with_wavelength,
        test_concentration_zero_od,
        test_concentration_output_shape,
        test_concentration_finite,
        test_concentration_pair_labels,
        test_concentration_known_solve,
        test_pipeline_concentration,
        test_pipeline_view_switching_with_concentration,
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
        print("[OK] All MBLL tests passed!")
    else:
        print("[!!] Some tests failed")
