"""
DataManager — Central data hub for the application.

Holds the currently loaded SNIRFData and emits Qt signals when the
data changes so that all GUI panels can react.
"""
from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal

from data_io.snirf_loader_base import SNIRFData
from data_io.snirf_loader_lib import SNIRFLoaderLib
from data_io.snirf_loader_h5py import SNIRFLoaderH5py


class DataManager(QObject):
    """Manages loaded fNIRS data and notifies listeners on changes."""

    # Signals
    data_loaded = pyqtSignal(object)    # emits SNIRFData
    data_cleared = pyqtSignal()
    error_occurred = pyqtSignal(str)    # emits error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: SNIRFData | None = None
        # Default to h5py loader (more reliable); can be swapped
        self._loader = SNIRFLoaderH5py()

    # ── Public API ────────────────────────────

    @property
    def data(self) -> SNIRFData | None:
        return self._data

    @property
    def has_data(self) -> bool:
        return self._data is not None

    def use_loader(self, loader_name: str):
        """Switch between 'snirf-library' and 'h5py-raw'."""
        if loader_name == "h5py-raw":
            self._loader = SNIRFLoaderH5py()
        else:
            self._loader = SNIRFLoaderLib()

    def load_file(self, filepath: str) -> bool:
        """
        Load a SNIRF file.  Returns True on success.
        Emits `data_loaded` on success or `error_occurred` on failure.
        """
        try:
            self._data = self._loader.load(filepath)
            self.data_loaded.emit(self._data)
            return True
        except Exception as exc:
            self._data = None
            self.error_occurred.emit(str(exc))
            return False

    def clear(self):
        self._data = None
        self.data_cleared.emit()
