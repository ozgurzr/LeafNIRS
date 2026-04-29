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

#### Block Averaging ✅

24. **Stimulus Parsing** — parse onset/duration/amplitude from SNIRF `stim` groups
25. **Epoch Extraction** — configurable pre/post windows, baseline correction
26. **Block Averaging** — trial-averaged HRF with SEM confidence bands
27. **Epoch Viewer GUI** — condition/pair selectors, HbO (red) / HbR (blue) HRF plot
28. **Stimulus Markers** — color-coded vertical onset lines on main time-series (toggleable)
29. **Epoch Tests** — 10 unit tests (extraction, baseline, averaging, pipeline)

#### Statistical Analysis — GLM ✅

30. **Canonical HRF** — double-gamma SPM model (scipy gamma PDF)
31. **Design Matrix** — stimulus convolution + polynomial drift regressors
32. **GLM Solver** — OLS fit, β weights, t-statistics, p-values per pair per condition
33. **Pipeline Integration** — `run_glm()` in ProcessingPipeline
34. **Probe Activation Map** — 2D scatter of t-stats on probe geometry (pyqtgraph)
35. **Run GLM Button** — GUI button with condition/chromophore/p-threshold controls
36. **GLM Tests** — 17 unit tests (HRF, design matrix, solver, integration)

#### HOMER3 Validation ✅

37. **Reference Pipeline** — MATLAB script generating 5 .mat reference outputs
38. **Cross-Validation** — Python comparison script: r² and RMSE at each stage
39. **Results** — 4/4 stages PASS (OD r²=1.00, filter r²=0.96, MBLL r²=0.96, blockavg r²=0.99)

### 5. Phase 4: 3D Brain Map Visualization ✅

40. **Brain Mesh Generator** — parametric ellipsoid with cortical folding + longitudinal fissure
41. **2D→3D Projection** — map probe 2D positions onto brain surface (nearest-surface lookup)
42. **3D Viewer Widget** — pyqtgraph GLViewWidget with translucent cortical mesh
43. **Activation Overlay** — GLM t-statistics as color-coded spheres at S-D midpoints
44. **Interactive Controls** — rotate/zoom, condition/chromophore/p-threshold selectors
45. **Tab UI** — tabbed panel: 📍 2D Probe Map | 🧠 3D Brain
46. **Dependency** — PyOpenGL added to requirements.txt

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
│   ├── epoch_extraction.py      # Block averaging + epoch extraction
│   ├── glm_analysis.py          # GLM: HRF, design matrix, OLS solver
│   ├── brain_mesh.py            # 3D brain mesh + 2D→3D projection
│   └── pipeline.py              # Processing state manager
├── gui/
│   ├── main_window.py           # Main app window
│   ├── file_info_panel.py       # Metadata display
│   ├── graph_widget.py          # PyQtGraph time-series
│   ├── processing_panel.py      # Auto/Manual controls
│   ├── epoch_viewer.py          # Block-averaged HRF viewer
│   ├── probe_map_widget.py      # 2D probe activation map
│   └── brain_viewer_widget.py   # 3D brain viewer with activation overlay
tests/
├── test_snirf_loaders.py        # 30 loader tests
├── test_processing.py           # 15 processing tests
├── test_mbll.py                 # 12 MBLL tests
├── test_motion_correction.py    # 11 motion correction tests
├── test_epoch_extraction.py     # 10 epoch tests
├── test_glm.py                  # 17 GLM tests
└── test_homer3_validation.py    # HOMER3 cross-validation
```
