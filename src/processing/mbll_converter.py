"""
Modified Beer-Lambert Law (MBLL) Converter.

Converts optical density (OD) changes into hemoglobin concentration
changes (ΔHbO, ΔHbR) using the two-wavelength MBLL approach:

    ΔOD(λ) = ε_HbO(λ) · ΔC_HbO · DPF(λ) · d  +  ε_HbR(λ) · ΔC_HbR · DPF(λ) · d

For two wavelengths this becomes a 2×2 linear system solved per
source-detector pair at each timepoint.
"""
from __future__ import annotations

import numpy as np

from data_io.snirf_loader_base import ChannelInfo, ProbeGeometry


# ══════════════════════════════════════════
#  Extinction Coefficients Table
# ══════════════════════════════════════════
#
# Molar extinction coefficients (cm⁻¹ / (mol/L)) for HbO₂ and HHb.
# Source: Scott Prahl (https://omlc.org/spectra/hemoglobin/)
# Values at common fNIRS wavelengths, converted to mm⁻¹/(mmol/L).
# Units: mm⁻¹ · L · mmol⁻¹  (i.e. per mmol/L per mm path)
#
# The table stores (ε_HbO, ε_HbR) for each wavelength in nm.
# These are the "specific extinction coefficients" commonly used
# in fNIRS literature (Cope, 1991; Strangman et al., 2003).

_EXTINCTION_TABLE: dict[int, tuple[float, float]] = {
    # nm:   (ε_HbO,     ε_HbR)     units: mm⁻¹/(mM)
    690:  (0.000956,  0.005186),
    700:  (0.001058,  0.004636),
    720:  (0.001260,  0.003600),
    730:  (0.001368,  0.003184),
    740:  (0.001516,  0.002804),
    750:  (0.001700,  0.002492),
    760:  (0.001896,  0.002236),
    770:  (0.002156,  0.002020),
    780:  (0.002500,  0.001844),
    790:  (0.002952,  0.001716),
    800:  (0.003452,  0.001618),   # isosbestic point
    810:  (0.003852,  0.001556),
    820:  (0.004204,  0.001516),
    830:  (0.004540,  0.001504),
    840:  (0.004852,  0.001504),
    850:  (0.005148,  0.001528),
    860:  (0.005424,  0.001560),
    870:  (0.005676,  0.001604),
    880:  (0.005916,  0.001660),
    890:  (0.006108,  0.001720),
    900:  (0.006264,  0.001800),
    910:  (0.006380,  0.001884),
    920:  (0.006464,  0.001988),
    940:  (0.006500,  0.002236),
    950:  (0.006424,  0.002384),
    960:  (0.006296,  0.002564),
    980:  (0.005924,  0.002988),
    1000: (0.005412,  0.003532),
}


def get_extinction_coefficients(
    wavelengths: np.ndarray | list[float],
) -> np.ndarray:
    """Look up extinction coefficients for given wavelengths.

    Parameters
    ----------
    wavelengths : array-like, shape (n_wl,)
        Wavelengths in nm.

    Returns
    -------
    E : ndarray, shape (n_wl, 2)
        Columns are [ε_HbO, ε_HbR] for each wavelength.

    Notes
    -----
    If exact wavelength is not in the table, nearest-neighbour
    interpolation is used from the available table entries.
    """
    wls = np.asarray(wavelengths, dtype=np.float64)
    table_wls = np.array(sorted(_EXTINCTION_TABLE.keys()), dtype=np.float64)
    E = np.zeros((len(wls), 2), dtype=np.float64)

    for i, wl in enumerate(wls):
        rounded = int(round(wl))
        if rounded in _EXTINCTION_TABLE:
            E[i] = _EXTINCTION_TABLE[rounded]
        else:
            # Nearest-neighbour from table
            idx = np.argmin(np.abs(table_wls - wl))
            nearest = int(table_wls[idx])
            E[i] = _EXTINCTION_TABLE[nearest]

    return E


# ══════════════════════════════════════════
#  Differential Pathlength Factor (DPF)
# ══════════════════════════════════════════
#
# DPF accounts for the increased optical path due to scattering.
# Formula from Scholkmann & Wolf (2013), General equation for DPF:
#   DPF(λ, age) = 223.3 + 0.05624 · age^0.8493
#                 - 5.723e-7 · λ^3 + 0.001245 · λ^2
#                 - 0.9025 · λ - 23.27 · exp(-0.0002051 · λ^2)
# (Simplified; common practice: use fixed DPF per wavelength.)

_DPF_TABLE: dict[int, float] = {
    # Common fNIRS wavelengths, adult head (age ~25)
    690: 6.51, 700: 6.40, 720: 6.20, 730: 6.10,
    740: 5.99, 750: 5.89, 760: 5.79, 770: 5.68,
    780: 5.57, 790: 5.46, 800: 5.36, 810: 5.26,
    820: 5.17, 830: 5.07, 840: 4.98, 850: 4.89,
    860: 4.81, 870: 4.73, 880: 4.65, 890: 4.58,
    900: 4.50, 910: 4.44, 920: 4.37, 940: 4.25,
    950: 4.19, 960: 4.13, 980: 4.03, 1000: 3.93,
}


def get_dpf(wavelengths: np.ndarray | list[float]) -> np.ndarray:
    """Look up differential pathlength factors for given wavelengths.

    Parameters
    ----------
    wavelengths : array-like, shape (n_wl,)
        Wavelengths in nm.

    Returns
    -------
    dpf : ndarray, shape (n_wl,)
        DPF value per wavelength (adult head, age ~25).
    """
    wls = np.asarray(wavelengths, dtype=np.float64)
    table_wls = np.array(sorted(_DPF_TABLE.keys()), dtype=np.float64)
    dpf = np.zeros(len(wls), dtype=np.float64)

    for i, wl in enumerate(wls):
        rounded = int(round(wl))
        if rounded in _DPF_TABLE:
            dpf[i] = _DPF_TABLE[rounded]
        else:
            idx = np.argmin(np.abs(table_wls - wl))
            nearest = int(table_wls[idx])
            dpf[i] = _DPF_TABLE[nearest]

    return dpf


# ══════════════════════════════════════════
#  OD → Concentration Conversion
# ══════════════════════════════════════════

def od_to_concentration(
    od: np.ndarray,
    channels: list[ChannelInfo],
    probe: ProbeGeometry,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert OD to hemoglobin concentration changes via MBLL.

    Groups channels by source-detector pair, then for each pair
    solves the 2×2 MBLL system using both wavelengths.

    Parameters
    ----------
    od : ndarray, shape (n_time, n_ch)
        Optical density (or filtered OD).
    channels : list of ChannelInfo
        Channel metadata (source/detector/wavelength indices).
    probe : ProbeGeometry
        Probe geometry (wavelengths, source/detector positions).

    Returns
    -------
    hbo : ndarray, shape (n_time, n_pairs)
        ΔHbO concentration changes (μmol/L).
    hbr : ndarray, shape (n_time, n_pairs)
        ΔHbR concentration changes (μmol/L).
    pair_labels : list of str
        Labels like 'S1-D1', 'S1-D2', etc.

    Notes
    -----
    - Requires exactly 2 wavelengths per S-D pair.
    - Source-detector distance is computed from probe positions.
    - Units: μmol/L (micromolar).
    """
    wavelengths = probe.wavelengths
    n_time = od.shape[0]

    # Get extinction coefficients and DPF
    E_raw = get_extinction_coefficients(wavelengths)  # (n_wl, 2)
    dpf = get_dpf(wavelengths)                        # (n_wl,)

    # Group channels by (source_index, detector_index) → {wl_idx: ch_idx}
    pair_map: dict[tuple[int, int], dict[int, int]] = {}
    for ch_idx, ch in enumerate(channels):
        key = (ch.source_index, ch.detector_index)
        pair_map.setdefault(key, {})[ch.wavelength_index] = ch_idx

    # Filter to pairs that have exactly 2 wavelengths
    valid_pairs = {
        k: v for k, v in pair_map.items()
        if len(v) == 2
    }

    n_pairs = len(valid_pairs)
    hbo = np.zeros((n_time, n_pairs), dtype=np.float64)
    hbr = np.zeros((n_time, n_pairs), dtype=np.float64)
    pair_labels: list[str] = []

    for pair_idx, ((src, det), wl_map) in enumerate(sorted(valid_pairs.items())):
        pair_labels.append(f"S{src}-D{det}")

        # Get the two channel indices and their wavelength indices
        wl_indices = sorted(wl_map.keys())  # e.g. [1, 2]
        ch1 = wl_map[wl_indices[0]]
        ch2 = wl_map[wl_indices[1]]

        # Compute source-detector distance (mm)
        src_pos = probe.source_pos[src - 1]  # 1-indexed → 0-indexed
        det_pos = probe.detector_pos[det - 1]
        sd_dist = np.linalg.norm(src_pos[:2] - det_pos[:2])  # 2D distance
        if sd_dist < 1.0:
            sd_dist = 30.0  # fallback: 30mm typical if geometry is missing

        # Build extinction matrix: E_matrix[i,j] = ε_j(λi) * DPF(λi) * d
        # i = wavelength index (0,1), j = chromophore (0=HbO, 1=HbR)
        wl_0 = wl_indices[0] - 1  # convert 1-indexed to 0-indexed
        wl_1 = wl_indices[1] - 1

        E_matrix = np.array([
            [E_raw[wl_0, 0] * dpf[wl_0] * sd_dist,
             E_raw[wl_0, 1] * dpf[wl_0] * sd_dist],
            [E_raw[wl_1, 0] * dpf[wl_1] * sd_dist,
             E_raw[wl_1, 1] * dpf[wl_1] * sd_dist],
        ])  # (2, 2)

        # Invert the extinction matrix
        try:
            E_inv = np.linalg.inv(E_matrix)
        except np.linalg.LinAlgError:
            # Singular matrix — skip this pair
            continue

        # OD for these two channels: (n_time, 2)
        od_pair = np.column_stack([od[:, ch1], od[:, ch2]])

        # Solve: [ΔC_HbO, ΔC_HbR] = E⁻¹ · [ΔOD(λ1), ΔOD(λ2)]ᵀ
        # → conc = od_pair @ E_inv.T  (each row is one timepoint)
        conc = od_pair @ E_inv.T  # (n_time, 2)

        hbo[:, pair_idx] = conc[:, 0]
        hbr[:, pair_idx] = conc[:, 1]

    return hbo, hbr, pair_labels
