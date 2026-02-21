"""DataManager — Holds loaded SNIRF data and coordinates loader selection."""
from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal

from data_io.snirf_loader_base import SNIRFData
from data_io.snirf_loader_lib import SNIRFLoaderLib
from data_io.snirf_loader_h5py import SNIRFLoaderH5py


class DataManager(QObject):
    """Manages the currently loaded SNIRF file and loader selection."""

    data_loaded = pyqtSignal(object)   # emits SNIRFData
    data_cleared = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: SNIRFData | None = None
        self._loaders = {
            'snirf-library': SNIRFLoaderLib(),
            'h5py-raw': SNIRFLoaderH5py(),
        }
        self._active_loader = 'h5py-raw'

    @property
    def data(self) -> SNIRFData | None:
        return self._data

    @property
    def active_loader_name(self) -> str:
        return self._active_loader

    def set_loader(self, name: str):
        if name in self._loaders:
            self._active_loader = name

    def load_file(self, filepath: str):
        try:
            loader = self._loaders[self._active_loader]
            self._data = loader.load(filepath)
            self.data_loaded.emit(self._data)
        except Exception as e:
            self._data = None
            self.error_occurred.emit(str(e))

    def close_file(self):
        self._data = None
        self.data_cleared.emit()
