# LeafNIRS — Project Plan & Progress

## Timeline & Phases

### 1. Proposal Preparation ✅ (Nov – Jan)

- Literature Review & Problem Definition
- System Architecture & Design
- Signal Processing Methodology
- Feasibility Prototype Development
- Proposal Drafting & Review
- **Milestone:** Final Proposal Submission (Early Jan)

### 2. Phase 1: Data Loading & GUI ✅ (Jan – Mar) — v0.1.0

1. **Project Setup** — folder structure, Python 3.12 venv, dependencies
2. **Dual SNIRF Loader** — `SNIRFLoaderBase` interface with two implementations:
   - `SNIRFLoaderH5py` (default, 2x faster) — raw HDF5 traversal
   - `SNIRFLoaderLib` (alternative) — `snirf` library wrapper
3. **Loader Tests** — 30 tests + 10 subtests, cross-validated both loaders
4. **GUI Main Window** — dark-themed PyQt5, file/view/help menus, status bar
5. **File Info Panel** — metadata display (channels, probe, wavelengths)
6. **Graph Widget** — PyQtGraph time-series with:
   - S-D pair grouping, wavelength filter (λ1/λ2/both)
   - Channel quality flags (CV-based: OK/flat/noisy)
   - Overlaid + stacked view modes, downsampling for 200+ ch files
7. **Data Manager** — Qt signal-based data hub

### 3. Phase 2: Signal Processing ✅ (Mar – Early Apr) — v0.2.0

8. **OD Conversion** — `od_converter.py`: OD = -log₁₀(I/I₀), epsilon clamping
9. **Bandpass Filter** — `bandpass_filter.py`: Butterworth via scipy filtfilt (zero-phase)
10. **Pipeline Manager** — `pipeline.py`: state tracking (RAW → OD → FILTERED), view switching
11. **Processing Panel** — `processing_panel.py`: Auto/Manual toggle, filter controls
12. **GUI Integration** — auto-apply on load, view switcher, reset, h5py default
13. **Processing Tests** — 15 unit tests (OD correctness, frequency response, pipeline state)
14. **MBLL Converter** — `mbll_converter.py`: extinction coefficients, DPF lookup, 2×2 solver
15. **Concentration Pipeline** — CONCENTRATION state, HbO/HbR storage in pipeline
16. **HbO/HbR Visualization** — red (HbO) / blue (HbR) curves per S-D pair
17. **MBLL Tests** — 12 unit tests (extinction, DPF, solver, pipeline integration)

### 4. Phase 3: Advanced Processing & Analysis (Apr – May)

#### Motion Artifact Correction ✅

18. **Artifact Detection** — temporal derivative + MAD threshold
19. **TDDR Correction** — Temporal Derivative Distribution Repair (Fishburn 2019)
20. **Spline Correction** — cubic spline interpolation over artifact segments
21. **CORRECTED Pipeline State** — placed between OD and FILTERED
22. **Apply All Button** — one-click full pipeline (OD → TDDR → Filter → MBLL)
23. **Motion Tests** — 11 unit tests (detection, TDDR, spline, pipeline)

#### Block Averaging ← *Next*

- Parse stimulus/event markers from SNIRF `stim` groups
- Epoch extraction around stimulus onsets
- Block averaging with confidence intervals
- GUI: stimulus markers on time-series, epoch viewer, averaged HRF plot

#### Statistical Analysis — GLM

- General Linear Model (GLM) for task-related activation
- t-maps / activation maps overlaid on probe geometry

#### HOMER3 Validation

- Cross-validate processing pipeline against HOMER3 outputs
- Benchmark OD, bandpass, MBLL, and block averaging results

### 5. Phase 4: 3D Brain Map Visualization (May – Jun)

- **Brain Atlas Integration** — MNI coordinate mapping
- **3D Rendering Engine** — cortical surface visualization
- **Interactive Overlay of HbO-HbR** — activation maps on 3D brain
- **GUI Integration** — 3D viewer panel with rotation/zoom

---

## Architecture Reference

```text
src/
├── core/
│   ├── data_manager.py          # Data hub, Qt signals
│   └── config_manager.py        # App settings
├── data_io/
│   ├── snirf_loader_base.py     # SNIRFData, ChannelInfo, ProbeInfo, StimulusInfo
│   ├── snirf_loader_h5py.py     # Method B (default)
│   └── snirf_loader_lib.py      # Method A (alternative)
├── processing/
│   ├── od_converter.py          # Intensity → OD
│   ├── bandpass_filter.py       # Butterworth bandpass
│   ├── motion_correction.py     # Artifact detection + TDDR/spline
│   ├── mbll_converter.py        # OD → HbO/HbR concentrations
│   └── pipeline.py              # Processing state manager
├── gui/
│   ├── main_window.py           # Main app window
│   ├── file_info_panel.py       # Metadata display
│   ├── graph_widget.py          # PyQtGraph time-series
│   └── processing_panel.py      # Auto/Manual controls
tests/
├── test_snirf_loaders.py        # 30 loader tests
├── test_processing.py           # 15 processing tests
├── test_mbll.py                 # 12 MBLL tests
└── test_motion_correction.py    # 11 motion correction tests
```
