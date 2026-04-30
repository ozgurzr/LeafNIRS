"""Batch group-level GLM analysis for LeafNIRS.

Usage:
    python scripts/batch_group_analysis.py --data-root ./data --output ./group_results

Discovers all .snirf files under --data-root, runs the full pipeline on each,
then performs second-level random-effects GLM across subjects.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_io.snirf_loader_h5py import SNIRFLoaderH5py
from processing.pipeline import ProcessingPipeline
from processing.glm_analysis import compute_glm
from processing.group_glm import run_group_glm
from processing.export_manager import export_csv, export_matlab


def parse_args():
    parser = argparse.ArgumentParser(description="LeafNIRS group-level GLM analysis")
    parser.add_argument("--data-root", required=True, help="Root directory containing .snirf files")
    parser.add_argument("--output", default="./group_results", help="Output directory")
    parser.add_argument("--filter-low", type=float, default=0.01, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--filter-high", type=float, default=0.1, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--filter-order", type=int, default=3, help="Filter order")
    return parser.parse_args()


def discover_snirf_files(root: str) -> list[Path]:
    """Recursively find all .snirf files under root."""
    return sorted(Path(root).rglob("*.snirf"))


def process_subject(snirf_path: Path, filter_low, filter_high, filter_order):
    """Run full pipeline + GLM on a single file. Returns (glm_hbo, glm_hbr) or None."""
    loader = SNIRFLoaderH5py()
    data = loader.load(str(snirf_path))

    if not data.stimuli:
        print(f"  SKIP (no stimuli): {snirf_path.name}")
        return None

    pipe = ProcessingPipeline(
        data.intensity, data.sampling_rate,
        channels=data.channels, probe=data.probe,
    )
    pipe.apply_motion_correction(method='tddr')
    pipe.apply_bandpass(low=filter_low, high=filter_high, order=filter_order)
    pipe.convert_to_concentration()
    r = pipe.result

    glm_hbo, glm_hbr = pipe.run_glm(data.stimuli)

    print(f"  OK: {snirf_path.name} — {len(glm_hbo.pair_labels)} pairs, "
          f"{len(glm_hbo.contrast_names)} conditions")
    return glm_hbo, glm_hbr


def plot_group_results(group_hbo, group_hbr, out_dir: Path):
    """Generate a summary bar chart of group t-statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle(f"Group GLM (n={group_hbo.n_subjects} subjects)", color='#dcdcdc',
                 fontsize=14, fontweight='bold')

    for ax, group, chrom, color in [
        (axes[0], group_hbo, 'HbO', '#e06c75'),
        (axes[1], group_hbr, 'HbR', '#61afef'),
    ]:
        ax.set_facecolor('#252526')
        for spine in ax.spines.values():
            spine.set_color('#3e3e42')
        ax.tick_params(colors='#dcdcdc', labelsize=8)
        ax.grid(True, alpha=0.1, axis='y', color='#888')

        n_conds = len(group.condition_names)
        n_pairs = len(group.pair_labels)
        x = np.arange(n_pairs)
        width = 0.8 / max(n_conds, 1)

        for ci in range(n_conds):
            t_vals = group.group_t_stat[ci, :]
            p_vals = group.group_p_value[ci, :]
            sig_mask = p_vals < 0.05

            offset = (ci - n_conds / 2 + 0.5) * width
            bars = ax.bar(x + offset, t_vals, width * 0.9, color=color, alpha=0.6,
                          label=group.condition_names[ci] if ci < 3 else None)

            # Mark significant pairs.
            for pi in range(n_pairs):
                if sig_mask[pi]:
                    ax.text(x[pi] + offset, t_vals[pi], '*',
                            ha='center', va='bottom', color='#e5c07b', fontsize=12)

        ax.set_xlabel('S-D Pair', color='#dcdcdc', fontsize=10)
        ax.set_ylabel('Group t-statistic', color='#dcdcdc', fontsize=10)
        ax.set_title(chrom, color=color, fontsize=12, fontweight='bold')
        ax.axhline(0, color='#888', linewidth=0.5)

        if n_pairs <= 20:
            ax.set_xticks(x)
            ax.set_xticklabels(group.pair_labels, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8, facecolor='#2d2d30', edgecolor='#3e3e42', labelcolor='#dcdcdc')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "group_glm_summary.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LeafNIRS Group-Level GLM Analysis")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output:    {out_dir}")
    print(f"Filter:    {args.filter_low}–{args.filter_high} Hz, order {args.filter_order}")
    print()

    snirf_files = discover_snirf_files(args.data_root)
    print(f"Found {len(snirf_files)} .snirf files\n")

    if not snirf_files:
        print("No .snirf files found. Exiting.")
        return

    hbo_results = []
    hbr_results = []
    subject_names = []

    for path in snirf_files:
        print(f"Processing: {path.name}")
        try:
            result = process_subject(path, args.filter_low, args.filter_high, args.filter_order)
            if result is not None:
                glm_hbo, glm_hbr = result
                hbo_results.append(glm_hbo)
                hbr_results.append(glm_hbr)
                subject_names.append(path.stem)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if len(hbo_results) < 2:
        print(f"\nOnly {len(hbo_results)} subjects processed. Need >= 2 for group analysis.")
        return

    print(f"\n{'─' * 40}")
    print(f"Running group GLM on {len(hbo_results)} subjects...")
    group_hbo = run_group_glm(hbo_results)
    group_hbr = run_group_glm(hbr_results)

    # Export group results.
    n_sig_hbo = int(np.sum(group_hbo.group_p_value < 0.05))
    n_sig_hbr = int(np.sum(group_hbr.group_p_value < 0.05))
    n_total = group_hbo.group_t_stat.size
    print(f"  HbO: {n_sig_hbo}/{n_total} significant (p<0.05)")
    print(f"  HbR: {n_sig_hbr}/{n_total} significant (p<0.05)")

    plot_path = plot_group_results(group_hbo, group_hbr, out_dir)
    print(f"  Plot: {plot_path}")

    # Save group CSV.
    import pandas as pd
    rows = []
    for ci, cond in enumerate(group_hbo.condition_names):
        for pi, pair in enumerate(group_hbo.pair_labels):
            rows.append({
                "condition": cond,
                "pair": pair,
                "hbo_group_t": group_hbo.group_t_stat[ci, pi],
                "hbo_group_p": group_hbo.group_p_value[ci, pi],
                "hbr_group_t": group_hbr.group_t_stat[ci, pi],
                "hbr_group_p": group_hbr.group_p_value[ci, pi],
                "n_subjects": group_hbo.n_subjects,
            })
    csv_path = out_dir / "group_glm_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  CSV:  {csv_path}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
