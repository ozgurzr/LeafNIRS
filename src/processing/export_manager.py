"""Export pipeline results to CSV and MATLAB formats."""
from __future__ import annotations

import os
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def export_csv(
    filepath: str,
    time: np.ndarray,
    data: np.ndarray,
    data_label: str,
    pair_labels: list[str] | None = None,
    channels: list | None = None,
    hbr: np.ndarray | None = None,
    glm_hbo=None,
    glm_hbr=None,
) -> list[str]:
    """Export current-view data and optional GLM results to CSV.

    Parameters
    ----------
    filepath : str
        Base path (without extension). Multiple files may be created.
    time : ndarray, shape (n_time,)
    data : ndarray, shape (n_time, n_cols)
        The active data array (raw, OD, filtered, or HbO).
    data_label : str
        Label for the data type (e.g., 'Raw Intensity', 'HbO / HbR').
    pair_labels : list of str, optional
        Column labels for concentration mode.
    channels : list of ChannelInfo, optional
        Channel metadata for raw/OD/filtered modes.
    hbr : ndarray, optional
        HbR array when exporting concentration data.
    glm_hbo, glm_hbr : GLMResult, optional
        GLM results to export as a second file.

    Returns
    -------
    created_files : list of str — paths of files written.
    """
    base = os.path.splitext(filepath)[0]
    created = []

    # --- Time-series file ---
    ts_path = f"{base}_timeseries.csv"
    df = _build_timeseries_df(time, data, data_label, pair_labels, channels, hbr)
    df.to_csv(ts_path, index=False, float_format="%.6f")
    created.append(ts_path)
    log.info("Exported time-series CSV: %s (%d rows)", ts_path, len(df))

    # --- GLM results file ---
    if glm_hbo is not None:
        glm_path = f"{base}_glm.csv"
        df_glm = _build_glm_df(glm_hbo, glm_hbr)
        df_glm.to_csv(glm_path, index=False, float_format="%.6f")
        created.append(glm_path)
        log.info("Exported GLM CSV: %s", glm_path)

    return created


def export_matlab(
    filepath: str,
    time: np.ndarray,
    data: np.ndarray,
    data_label: str,
    sampling_rate: float,
    pair_labels: list[str] | None = None,
    channels: list | None = None,
    hbr: np.ndarray | None = None,
    probe=None,
    glm_hbo=None,
    glm_hbr=None,
) -> str:
    """Export current-view data and optional GLM results to .mat file.

    Uses descriptive variable names: hbo, hbr, time, pair_labels, etc.

    Returns
    -------
    filepath : str — the .mat path written.
    """
    from scipy.io import savemat

    if not filepath.endswith(".mat"):
        filepath += ".mat"

    mdict = {
        "time": time.astype(np.float64),
        "data": data.astype(np.float64),
        "data_label": data_label,
        "sampling_rate": float(sampling_rate),
    }

    if pair_labels is not None:
        mdict["pair_labels"] = np.array(pair_labels, dtype=object)

    if hbr is not None:
        mdict["hbo"] = data.astype(np.float64)
        mdict["hbr"] = hbr.astype(np.float64)

    if probe is not None:
        mdict["probe_source_pos"] = np.asarray(probe.source_pos, dtype=np.float64)
        mdict["probe_detector_pos"] = np.asarray(probe.detector_pos, dtype=np.float64)
        mdict["wavelengths"] = np.asarray(probe.wavelengths, dtype=np.float64)

    if channels is not None:
        src_idx = np.array([ch.source_index for ch in channels], dtype=np.int32)
        det_idx = np.array([ch.detector_index for ch in channels], dtype=np.int32)
        wl_idx = np.array([ch.wavelength_index for ch in channels], dtype=np.int32)
        mdict["channel_source_index"] = src_idx
        mdict["channel_detector_index"] = det_idx
        mdict["channel_wavelength_index"] = wl_idx

    if glm_hbo is not None:
        mdict["glm_hbo_beta"] = glm_hbo.beta.astype(np.float64)
        mdict["glm_hbo_tstat"] = glm_hbo.t_stat.astype(np.float64)
        mdict["glm_hbo_pvalue"] = glm_hbo.p_value.astype(np.float64)
        mdict["glm_conditions"] = np.array(glm_hbo.contrast_names, dtype=object)
        mdict["glm_pair_labels"] = np.array(glm_hbo.pair_labels, dtype=object)
        mdict["design_matrix"] = glm_hbo.design_matrix.astype(np.float64)

    if glm_hbr is not None:
        mdict["glm_hbr_beta"] = glm_hbr.beta.astype(np.float64)
        mdict["glm_hbr_tstat"] = glm_hbr.t_stat.astype(np.float64)
        mdict["glm_hbr_pvalue"] = glm_hbr.p_value.astype(np.float64)

    savemat(filepath, mdict, do_compression=True)
    log.info("Exported MATLAB file: %s", filepath)
    return filepath


def _build_timeseries_df(
    time: np.ndarray,
    data: np.ndarray,
    data_label: str,
    pair_labels: list[str] | None,
    channels: list | None,
    hbr: np.ndarray | None,
) -> pd.DataFrame:
    """Build a DataFrame from the active data array."""
    is_conc = "HbO" in data_label or hbr is not None
    df = pd.DataFrame({"time_s": time})

    if is_conc and pair_labels is not None:
        n_pairs = data.shape[1]
        for i in range(n_pairs):
            label = pair_labels[i] if i < len(pair_labels) else f"pair_{i}"
            df[f"HbO_{label}"] = data[:, i]
            if hbr is not None:
                df[f"HbR_{label}"] = hbr[:, i]
    else:
        n_ch = data.shape[1] if data.ndim > 1 else 1
        for ch in range(n_ch):
            if channels and ch < len(channels):
                c = channels[ch]
                col = f"S{c.source_index}_D{c.detector_index}_wl{c.wavelength_index}"
            else:
                col = f"ch_{ch}"
            col_data = data[:, ch] if data.ndim > 1 else data
            df[col] = col_data

    return df


def _build_glm_df(glm_hbo, glm_hbr=None) -> pd.DataFrame:
    """Build a DataFrame with GLM statistics (beta, t, p) per condition and pair."""
    rows = []
    for ci, cond_name in enumerate(glm_hbo.contrast_names):
        for pi, pair_label in enumerate(glm_hbo.pair_labels):
            row = {
                "condition": cond_name,
                "pair": pair_label,
                "hbo_beta": glm_hbo.beta[ci, pi],
                "hbo_tstat": glm_hbo.t_stat[ci, pi],
                "hbo_pvalue": glm_hbo.p_value[ci, pi],
            }
            if glm_hbr is not None:
                row["hbr_beta"] = glm_hbr.beta[ci, pi]
                row["hbr_tstat"] = glm_hbr.t_stat[ci, pi]
                row["hbr_pvalue"] = glm_hbr.p_value[ci, pi]
            rows.append(row)

    return pd.DataFrame(rows)
