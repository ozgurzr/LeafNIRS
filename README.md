# LeafNIRS

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase 1](https://img.shields.io/badge/Phase-1%20Complete-brightgreen)](https://github.com/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-41CD52?logo=qt&logoColor=white)](https://pypi.org/project/PyQt5/)
[![SNIRF](https://img.shields.io/badge/Format-SNIRF%20%2F%20HDF5-orange)](https://fnirs.org/resources/software/snirf/)

A Python-based **fNIRS Brain Mapping Tool** for signal processing and visualization, supporting the [SNIRF](https://fnirs.org/resources/software/snirf/) standard.

> [!NOTE]
> This project is under active development as a senior design project at Acibadem Mehmet Ali Aydinlar University, Department of Biomedical Engineering.

![LeafNIRS Screenshot](docs/screenshot.png)

---

## Features

- SNIRF / HDF5 file loading with dual loader options
- Dark-themed interactive GUI with real-time channel plotting
- Source-detector pair grouping with wavelength filtering
- Automatic signal quality assessment
- Stacked and overlaid view modes

## Quick Start

```bash
git clone https://github.com/ozgurzr/LeafNIRS.git
cd LeafNIRS
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Then use **File → Open SNIRF…** to load a `.snirf` file.

## Running Tests

Tests require `.snirf` data files. Place any SNIRF dataset in a `fNIRS_1/` folder next to the repo:

```bash
python -m pytest tests/test_snirf_loaders.py -v
```

## Project Structure

```text
LeafNIRS/
├── run.py                          # Entry point
├── requirements.txt
├── src/
│   ├── data_io/                    # SNIRF loaders
│   │   ├── snirf_loader_base.py    # Abstract interface + data model
│   │   ├── snirf_loader_lib.py     # Method A: snirf library
│   │   └── snirf_loader_h5py.py    # Method B: raw h5py (2x faster)
│   ├── core/                       # Application logic
│   │   ├── data_manager.py         # Loader orchestration
│   │   └── config_manager.py       # User preferences
│   ├── gui/                        # PyQt5 interface
│   │   ├── main_window.py          # Main application window
│   │   ├── file_info_panel.py      # File metadata display
│   │   └── graph_widget.py         # Time-series viewer
│   └── processing/                 # Signal processing (Phase 2)
├── tests/
│   └── test_snirf_loaders.py       # Loader test suite
└── docs/
    └── phase1_notes.md             # Development notes
```

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Data loading & basic visualization | ✅ Complete |
| **2** | Bandpass filtering & signal processing | 🔜 Planned |
| **3** | Modified Beer-Lambert Law (HbO / HbR) | 🔜 Planned |
| **4** | 3D brain mapping & topographic display | 🔜 Planned |
| **5** | Statistical analysis & export | 🔜 Planned |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.
