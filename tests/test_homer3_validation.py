"""HOMER3 cross-validation: compare LeafNIRS pipeline outputs against HOMER3 reference."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.io import loadmat

from data_io.snirf_loader_h5py import SNIRFLoaderH5py
from processing.pipeline import ProcessingPipeline, PipelineState
from processing.epoch_extraction import compute_condition_average

SNIRF_FILE = r"C:\Users\90546\Desktop\LeafNIRS Plan\data\DATA_MotorNeuron\SUBJID_6082\run1.snirf"
REF_DIR = r"C:\Users\90546\Desktop\LeafNIRS Plan\data\homer3_reference"


def load_ref(name):
    return loadmat(os.path.join(REF_DIR, name))


def correlation(a, b):
    """Pearson r² between two flattened arrays."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    a_flat, b_flat = a_flat[mask], b_flat[mask]
    if len(a_flat) == 0:
        return 0.0
    r = np.corrcoef(a_flat, b_flat)[0, 1]
    return r ** 2


def rmse(a, b):
    a_flat = a.ravel()
    b_flat = b.ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    a_flat, b_flat = a_flat[mask], b_flat[mask]
    return np.sqrt(np.mean((a_flat - b_flat) ** 2))


def print_comparison(stage, ours, homer, threshold_r2=0.95):
    r2 = correlation(ours, homer)
    err = rmse(ours, homer)
    scale = np.std(homer.ravel()[np.isfinite(homer.ravel())])
    nrmse = err / scale if scale > 0 else float('inf')
    status = "PASS" if r2 >= threshold_r2 else "FAIL"
    print(f"  [{status}] {stage}")
    print(f"         r² = {r2:.6f}   RMSE = {err:.6e}   nRMSE = {nrmse:.4f}")
    print(f"         Ours:  shape={ours.shape}  range=[{np.nanmin(ours):.6f}, {np.nanmax(ours):.6f}]")
    print(f"         HOMER: shape={homer.shape}  range=[{np.nanmin(homer):.6f}, {np.nanmax(homer):.6f}]")
    return status == "PASS"


def main():
    print("=" * 70)
    print("  LeafNIRS vs HOMER3 Cross-Validation")
    print("=" * 70)

    # Load SNIRF
    loader = SNIRFLoaderH5py()
    data = loader.load(SNIRF_FILE)
    print(f"\nLoaded: {data.n_channels} ch, {data.n_timepoints} timepoints, {data.sampling_rate} Hz")

    results = []

    # ── Stage 1: OD ──
    print("\n--- Stage 1: Optical Density ---")
    pipe = ProcessingPipeline(data.intensity, data.sampling_rate,
                              channels=data.channels, probe=data.probe)
    our_od = pipe.convert_to_od()

    ref = load_ref('homer3_od.mat')
    homer_od = ref['dod_matrix']
    results.append(print_comparison("OD Conversion", our_od, homer_od))

    # ── Stage 2: Bandpass Filter (skip TDDR to match HOMER3) ──
    print("\n--- Stage 2: Bandpass Filter (0.01–0.1 Hz) ---")
    print("  (TDDR skipped in HOMER3, so we also skip it for fair comparison)")

    # Reset and redo without TDDR to match HOMER3's pipeline
    pipe_no_tddr = ProcessingPipeline(data.intensity, data.sampling_rate,
                                      channels=data.channels, probe=data.probe)
    pipe_no_tddr.convert_to_od()
    our_filtered = pipe_no_tddr.apply_bandpass(low=0.01, high=0.1, order=3)

    ref = load_ref('homer3_filtered.mat')
    homer_filtered = ref['dod_filt_matrix']
    results.append(print_comparison("Bandpass Filter", our_filtered, homer_filtered, threshold_r2=0.90))

    # ── Stage 3: MBLL (HbO / HbR) ──
    print("\n--- Stage 3: MBLL Concentration ---")
    our_hbo_raw, our_hbr_raw = pipe_no_tddr.convert_to_concentration()

    ref = load_ref('homer3_conc.mat')
    homer_conc = ref['dc_matrix']
    # HOMER3 dc_matrix is [time, 3*n_pairs]: columns are HbO, HbR, HbT interleaved
    n_pairs = our_hbo_raw.shape[1]
    homer_hbo = homer_conc[:, 0::3][:, :n_pairs]  # every 3rd column starting at 0
    homer_hbr = homer_conc[:, 1::3][:, :n_pairs]  # every 3rd column starting at 1

    # HOMER3 outputs in mol/L (or mmol/L), LeafNIRS may use µmol/L
    # Check scale difference
    our_hbo_std = np.std(our_hbo_raw)
    homer_hbo_std = np.std(homer_hbo)
    if our_hbo_std > 0 and homer_hbo_std > 0:
        scale_ratio = our_hbo_std / homer_hbo_std
        print(f"  Scale ratio (ours/HOMER3): {scale_ratio:.2f}")
        if scale_ratio > 100:
            print(f"  -> HOMER3 is in mol/L, ours in µmol/L. Scaling HOMER3 × 1e6")
            homer_hbo = homer_hbo * 1e6
            homer_hbr = homer_hbr * 1e6
        elif scale_ratio > 1.5:
            print(f"  -> HOMER3 is in mmol/L, ours in µmol/L. Scaling HOMER3 × 1e3")
            homer_hbo = homer_hbo * 1e3
            homer_hbr = homer_hbr * 1e3

    print("\n  HbO:")
    r_hbo = print_comparison("MBLL - HbO", our_hbo_raw, homer_hbo)
    print("\n  HbR:")
    r_hbr = print_comparison("MBLL - HbR", our_hbr_raw, homer_hbr)
    results.append(r_hbo and r_hbr)

    # ── Stage 4: Block Average ──
    print("\n--- Stage 4: Block Average (Condition 1, [-2, 20]s) ---")
    stim = data.stimuli[0]
    pair_labels = pipe_no_tddr.result.pair_labels

    ba_result = compute_condition_average(
        hbo=our_hbo_raw, hbr=our_hbr_raw,
        time=data.time,
        onsets=stim.onset,
        condition_name=stim.name,
        pair_labels=pair_labels,
        pre_sec=2.0, post_sec=20.0,
    )

    ref = load_ref('homer3_blockavg.mat')
    homer_bavg = ref['dcAvg_matrix']
    homer_tHRF = ref['tHRF'].ravel()

    # HOMER3 block avg has all conditions × all chromophores interleaved
    # For condition 1 (first block of 3*n_pairs columns): HbO=0::3, HbR=1::3
    cond1_cols = 3 * n_pairs  # columns for condition 1
    homer_bavg_hbo = homer_bavg[:, 0:cond1_cols:3][:, :n_pairs]
    homer_bavg_hbr = homer_bavg[:, 1:cond1_cols:3][:, :n_pairs]

    # Match time lengths
    min_t = min(ba_result.hbo_mean.shape[0], homer_bavg_hbo.shape[0])
    our_ba_hbo = ba_result.hbo_mean[:min_t, :]
    our_ba_hbr = ba_result.hbr_mean[:min_t, :]
    homer_ba_hbo = homer_bavg_hbo[:min_t, :]
    homer_ba_hbr = homer_bavg_hbr[:min_t, :]

    # Apply same scale correction
    if scale_ratio > 100:
        homer_ba_hbo = homer_ba_hbo * 1e6
        homer_ba_hbr = homer_ba_hbr * 1e6
    elif scale_ratio > 1.5:
        homer_ba_hbo = homer_ba_hbo * 1e3
        homer_ba_hbr = homer_ba_hbr * 1e3

    print(f"  Trials: ours={ba_result.n_trials}, HOMER3 excluded 1 trial")
    print(f"  Time points compared: {min_t}")
    print("\n  Block Avg HbO:")
    r_ba_hbo = print_comparison("BlockAvg - HbO", our_ba_hbo, homer_ba_hbo, threshold_r2=0.80)
    print("\n  Block Avg HbR:")
    r_ba_hbr = print_comparison("BlockAvg - HbR", our_ba_hbr, homer_ba_hbr, threshold_r2=0.80)
    results.append(r_ba_hbo and r_ba_hbr)

    # ── Summary ──
    print("\n" + "=" * 70)
    stages = ["OD Conversion", "Bandpass Filter", "MBLL (HbO+HbR)", "Block Average"]
    passed = sum(results)
    for i, (stage, ok) in enumerate(zip(stages, results)):
        print(f"  {'PASS' if ok else 'FAIL'}  {stage}")
    print(f"\n  Result: {passed}/{len(results)} stages passed")
    if passed == len(results):
        print("  VALIDATION PASSED - LeafNIRS matches HOMER3")
    else:
        print("  VALIDATION PARTIAL - Check failed stages above")
    print("=" * 70)


if __name__ == "__main__":
    main()
