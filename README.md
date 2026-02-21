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

- **Dual SNIRF Loader** — Load `.snirf` files via the `snirf` library (Method A) or raw `h5py` (Method B, 2x faster)
- **Dark-Themed GUI** — Professional PyQt5 interface with interactive PyQtGraph plotting
- **Channel Grouping** — Channels organized by source-detector pair with per-wavelength toggles
- **Wavelength Filter** — Instantly filter by 760 nm, 850 nm, or both
- **Signal Quality Assessment** — Automatic CV-based flagging of OK, flat, and noisy channels
- **Stacked / Overlaid Views** — Compare channel waveforms side-by-side or overlaid
- **Comprehensive Test Suite** — 30+ tests validated against real OpenNeuro ds007420 data

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/LeafNIRS.git
cd LeafNIRS

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Launch
python run.py
```

Then use **File → Open SNIRF…** to load a `.snirf` file.

## Running Tests

Tests require real SNIRF data from [OpenNeuro ds007420](https://openneuro.org/datasets/ds007420). Place the data in a `fNIRS_1/` folder next to `LeafNIRS/`:

```
parent_folder/
├── LeafNIRS/       ← this repo
└── fNIRS_1/        ← SNIRF data here
    └── sub-170/
        └── ses-01/
            └── nirs/
                └── *.snirf
```

```bash
python -m pytest tests/test_snirf_loaders.py -v
```

## Project Structure

```
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

## Team

| Role | Name |
|------|------|
| Developer | Ali Umut Sezgin |
| Developer | Arda Telci |
| Developer | Özgür Efe Zurnaci |
| Supervisor | Prof. Dr. Ata Akin |

**Acibadem Mehmet Ali Aydinlar University** — Department of Biomedical Engineering

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.
