"""SNIRF Loader — Abstract interface and unified data model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProbeGeometry:
    """Source/detector positions and wavelengths."""
    source_pos: np.ndarray       # (n_src, 2|3)
    detector_pos: np.ndarray     # (n_det, 2|3)
    wavelengths: np.ndarray      # (n_wl,)
    source_labels: list[str] = field(default_factory=list)
    detector_labels: list[str] = field(default_factory=list)


@dataclass
class ChannelInfo:
    """Per-channel measurement metadata."""
    source_index: int
    detector_index: int
    wavelength_index: int
    data_type: int
    data_type_label: str = ""


@dataclass
class StimulusInfo:
    """Single stimulus condition."""
    name: str
    onset: np.ndarray
    duration: np.ndarray
    amplitude: np.ndarray


@dataclass
class SNIRFData:
    """Unified container for all data extracted from a SNIRF file."""
    intensity: np.ndarray            # (n_time, n_ch)
    time: np.ndarray                 # (n_time,)
    probe: ProbeGeometry
    channels: list[ChannelInfo]
    stimuli: list[StimulusInfo] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    filepath: str = ""

    @property
    def n_channels(self) -> int:
        return self.intensity.shape[1] if self.intensity.ndim == 2 else 0

    @property
    def n_timepoints(self) -> int:
        return self.intensity.shape[0]

    @property
    def duration_seconds(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    @property
    def sampling_rate(self) -> float:
        if len(self.time) < 2:
            return 0.0
        return 1.0 / float(np.median(np.diff(self.time)))

    @property
    def n_sources(self) -> int:
        return self.probe.source_pos.shape[0]

    @property
    def n_detectors(self) -> int:
        return self.probe.detector_pos.shape[0]

    @property
    def wavelength_list(self) -> list[float]:
        return self.probe.wavelengths.tolist()


class SNIRFLoaderBase(ABC):
    """Abstract base class for SNIRF file loaders."""

    def load(self, filepath: str) -> SNIRFData:
        import os
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        if not filepath.lower().endswith('.snirf'):
            raise ValueError(f"Not a SNIRF file: {filepath}")
        data = self._load_impl(filepath)
        data.filepath = filepath
        return data

    @abstractmethod
    def _load_impl(self, filepath: str) -> SNIRFData: ...

    @staticmethod
    def loader_name() -> str:
        return "base"
