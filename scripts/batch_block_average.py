"""Batch block-average computation for LeafNIRS — multiple subjects/runs.

Produces:
  - CSV files with block-averaged HbO/HbR per condition per run
  - Publication-quality plots of HRF per condition
  - Summary log for the professor
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from data_io.snirf_loader_h5py import SNIRFLoaderH5py
from processing.pipeline import ProcessingPipeline
from processing.epoch_extraction import compute_condition_average

# ── Configuration ──
DATA_ROOT = Path(r"C:\Users\90546\Desktop\LeafNIRS Plan\data\DATA_MotorNeuron")
OUT_DIR = Path(r"C:\Users\90546\Desktop\LeafNIRS Plan\LeafNIRS\block_average_results")
OUT_DIR.mkdir(exist_ok=True)

SUBJECTS = ["SUBJID_6082", "SUBJID_6083", "SUBJID_6084"]
RUNS = ["run1", "run2", "run3"]

PRE_SEC = 2.0
POST_SEC = 20.0

FILTER_LOW = 0.01
FILTER_HIGH = 0.1
FILTER_ORDER = 3

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def process_run(subj, run):
    """Process a single run and return block-average results per condition."""
    snirf_path = DATA_ROOT / subj / f"{run}.snirf"
    if not snirf_path.exists():
        log(f"  SKIP: {snirf_path} not found")
        return None

    loader = SNIRFLoaderH5py()
    data = loader.load(str(snirf_path))

    log(f"  Loaded: {data.n_channels} ch, {data.n_timepoints} tp, "
        f"{data.sampling_rate:.1f} Hz, {len(data.stimuli)} conditions")

    # Full pipeline: OD → TDDR → Bandpass → MBLL
    pipe = ProcessingPipeline(
        data.intensity, data.sampling_rate,
        channels=data.channels, probe=data.probe,
    )
    pipe.apply_motion_correction(method='tddr')
    pipe.apply_bandpass(low=FILTER_LOW, high=FILTER_HIGH, order=FILTER_ORDER)
    pipe.convert_to_concentration()
    r = pipe.result

    log(f"  Pipeline: {r.hbo.shape[1]} pairs, HbO shape={r.hbo.shape}")

    # Block average per condition
    results = []
    for stim in data.stimuli:
        ba = compute_condition_average(
            hbo=r.hbo, hbr=r.hbr, time=data.time,
            onsets=stim.onset, condition_name=stim.name,
            pair_labels=r.pair_labels,
            pre_sec=PRE_SEC, post_sec=POST_SEC,
        )
        results.append(ba)
        log(f"    Cond '{stim.name}': {ba.n_trials} trials, "
            f"epoch_time={ba.epoch_time[0]:.1f}..{ba.epoch_time[-1]:.1f}s")

    return results


def save_csv(ba, subj, run, out_dir):
    """Save block-average as CSV."""
    fname = f"{subj}_{run}_cond{ba.condition}_blockavg.csv"
    path = out_dir / fname

    n_time = len(ba.epoch_time)
    n_pairs = ba.hbo_mean.shape[1] if ba.hbo_mean.ndim > 1 else 1

    with open(path, 'w') as f:
        # Header
        cols = ["time_s"]
        for i, label in enumerate(ba.pair_labels[:n_pairs]):
            cols.append(f"HbO_mean_{label}")
            cols.append(f"HbO_sem_{label}")
            cols.append(f"HbR_mean_{label}")
            cols.append(f"HbR_sem_{label}")
        f.write(",".join(cols) + "\n")

        # Data rows
        for t in range(n_time):
            row = [f"{ba.epoch_time[t]:.4f}"]
            for p in range(n_pairs):
                row.append(f"{ba.hbo_mean[t, p]:.6f}")
                row.append(f"{ba.hbo_sem[t, p]:.6f}")
                row.append(f"{ba.hbr_mean[t, p]:.6f}")
                row.append(f"{ba.hbr_sem[t, p]:.6f}")
            f.write(",".join(row) + "\n")

    return fname


def plot_hrf(ba, subj, run, out_dir):
    """Plot grand-average HRF (mean across all pairs) for one condition."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#252526')

    # Grand-average across pairs
    hbo_grand = ba.hbo_mean.mean(axis=1)
    hbo_sem_grand = ba.hbo_sem.mean(axis=1)
    hbr_grand = ba.hbr_mean.mean(axis=1)
    hbr_sem_grand = ba.hbr_sem.mean(axis=1)
    t = ba.epoch_time

    ax.fill_between(t, hbo_grand - hbo_sem_grand, hbo_grand + hbo_sem_grand,
                    alpha=0.25, color='#e06c75')
    ax.plot(t, hbo_grand, color='#e06c75', linewidth=2, label='HbO')

    ax.fill_between(t, hbr_grand - hbr_sem_grand, hbr_grand + hbr_sem_grand,
                    alpha=0.25, color='#61afef')
    ax.plot(t, hbr_grand, color='#61afef', linewidth=2, label='HbR')

    ax.axvline(x=0, color='#e5c07b', linestyle='--', linewidth=1, alpha=0.7, label='Onset')
    ax.axhline(y=0, color='#888', linestyle='-', linewidth=0.5, alpha=0.4)

    ax.set_xlabel('Time (s)', color='#dcdcdc', fontsize=11)
    ax.set_ylabel('d[Concentration] (umol/L)', color='#dcdcdc', fontsize=11)
    ax.set_title(
        f'{subj} — {run} — Condition {ba.condition}\n'
        f'Block Average (n={ba.n_trials} trials, {len(ba.pair_labels)} pairs)',
        color='#dcdcdc', fontsize=12, fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=10, facecolor='#2d2d30',
              edgecolor='#3e3e42', labelcolor='#dcdcdc')
    ax.tick_params(colors='#dcdcdc')
    for spine in ax.spines.values():
        spine.set_color('#3e3e42')
    ax.grid(True, alpha=0.15, color='#888')

    fname = f"{subj}_{run}_cond{ba.condition}_hrf.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


def plot_summary_grid(all_results, out_dir):
    """Plot a summary grid: rows=subjects, cols=conditions."""
    conditions = sorted(set(
        ba.condition for subj_results in all_results.values()
        for run_results in subj_results.values()
        for ba in run_results
    ))
    subjects = sorted(all_results.keys())
    n_conds = len(conditions)
    n_subjs = len(subjects)

    fig, axes = plt.subplots(n_subjs, n_conds, figsize=(4 * n_conds, 3 * n_subjs),
                             squeeze=False)
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle('Block-Averaged HRF — Grand Mean Across Runs',
                 color='#dcdcdc', fontsize=14, fontweight='bold', y=0.98)

    for si, subj in enumerate(subjects):
        for ci, cond in enumerate(conditions):
            ax = axes[si, ci]
            ax.set_facecolor('#252526')
            for spine in ax.spines.values():
                spine.set_color('#3e3e42')
            ax.tick_params(colors='#888', labelsize=8)
            ax.grid(True, alpha=0.1, color='#888')

            hbo_runs = []
            hbr_runs = []
            epoch_time = None
            for run, results in all_results[subj].items():
                for ba in results:
                    if ba.condition == cond:
                        hbo_runs.append(ba.hbo_mean.mean(axis=1))
                        hbr_runs.append(ba.hbr_mean.mean(axis=1))
                        epoch_time = ba.epoch_time

            if not hbo_runs or epoch_time is None:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', color='#888')
                continue

            hbo_all = np.array(hbo_runs)
            hbr_all = np.array(hbr_runs)
            t = epoch_time

            for i in range(len(hbo_runs)):
                ax.plot(t, hbo_runs[i], color='#e06c75', alpha=0.3, linewidth=0.8)
                ax.plot(t, hbr_runs[i], color='#61afef', alpha=0.3, linewidth=0.8)

            ax.plot(t, hbo_all.mean(axis=0), color='#e06c75', linewidth=2)
            ax.plot(t, hbr_all.mean(axis=0), color='#61afef', linewidth=2)
            ax.axvline(0, color='#e5c07b', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.axhline(0, color='#888', linewidth=0.3)

            if si == 0:
                ax.set_title(f'Cond {cond}', color='#dcdcdc', fontsize=10, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(subj.replace('SUBJID_', 'S'),
                              color='#dcdcdc', fontsize=9)
            if si == n_subjs - 1:
                ax.set_xlabel('Time (s)', color='#888', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "summary_grid_all_subjects.png"
    fig.savefig(out_dir / fname, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


def main():
    log("=" * 60)
    log("LeafNIRS Block-Average Batch Processing")
    log("=" * 60)
    log(f"Pipeline: OD -> TDDR -> Bandpass ({FILTER_LOW}-{FILTER_HIGH} Hz) -> MBLL")
    log(f"Epoch window: -{PRE_SEC}s to +{POST_SEC}s")
    log(f"Output: {OUT_DIR}")
    log("")

    all_results = {}
    all_files = []

    for subj in SUBJECTS:
        log(f"\n{'─'*40}")
        log(f"Subject: {subj}")
        all_results[subj] = {}

        for run in RUNS:
            log(f"\n  Run: {run}")
            results = process_run(subj, run)
            if results is None:
                continue
            all_results[subj][run] = results

            for ba in results:
                csv_name = save_csv(ba, subj, run, OUT_DIR)
                png_name = plot_hrf(ba, subj, run, OUT_DIR)
                all_files.append(csv_name)
                all_files.append(png_name)
                log(f"    Saved: {csv_name}")
                log(f"    Saved: {png_name}")

    # Summary grid
    log(f"\n{'─'*40}")
    log("Generating summary grid...")
    grid_name = plot_summary_grid(all_results, OUT_DIR)
    all_files.append(grid_name)
    log(f"  Saved: {grid_name}")

    # Summary statistics
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    total_runs = sum(len(runs) for runs in all_results.values())
    total_ba = sum(
        len(results)
        for subj_runs in all_results.values()
        for results in subj_runs.values()
    )
    log(f"Subjects: {len(SUBJECTS)}")
    log(f"Total runs processed: {total_runs}")
    log(f"Total block averages: {total_ba}")
    log(f"Files generated: {len(all_files)}")
    log("")
    log("Files:")
    for f in sorted(all_files):
        log(f"  {f}")

    # Save log
    log_path = OUT_DIR / "processing_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
