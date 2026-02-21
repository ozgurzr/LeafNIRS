# Contributing to LeafNIRS

Thank you for your interest in contributing to LeafNIRS! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run the test suite: `python -m pytest tests/ -v`
6. Submit a pull request

## Project Architecture

LeafNIRS follows a layered architecture:

```
GUI (PyQt5)  →  Core Logic  →  Data I/O (SNIRF loaders)
                     ↓
              Signal Processing
```

- **`src/data_io/`** — SNIRF file loaders. Both loaders implement the `SNIRFLoaderBase` interface.
- **`src/core/`** — Application state management and configuration.
- **`src/gui/`** — PyQt5 widgets and the main application window.
- **`src/processing/`** — Signal processing pipeline (Phase 2+).

## Coding Standards

- **Python 3.12+** with type hints
- **Docstrings** on all public classes and functions
- **PEP 8** style (enforced via linter)
- **No hardcoded paths** — use `os.path` or `pathlib`

## Testing

- Tests live in `tests/` and use `pytest`
- Real data tests require OpenNeuro ds007420 `.snirf` files in `../fNIRS_1/`
- Tests will be skipped automatically if test data is not present

## Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error traceback if applicable
