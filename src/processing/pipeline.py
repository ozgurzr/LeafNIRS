"""
Processing Pipeline Manager.

Orchestrates the signal processing stages and tracks the current
processing state. Stores intermediate results so the user can
switch between views (Raw / OD / Filtered).
"""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np

from processing.od_converter import intensity_to_od
from processing.bandpass_filter import bandpass_filter


class PipelineState(Enum):
    """Current stage of the processing pipeline."""
    RAW = auto()
    OD = auto()
    FILTERED = auto()


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    raw: np.ndarray              # (n_time, n_ch) original intensity
    od: np.ndarray | None = None  # (n_time, n_ch) optical density
    filtered: np.ndarray | None = None  # (n_time, n_ch) bandpass filtered OD
    state: PipelineState = PipelineState.RAW

    # Filter parameters used (for display)
    filter_low: float = 0.0
    filter_high: float = 0.0
    filter_order: int = 0

    @property
    def active_data(self) -> np.ndarray:
        """Return the data array for the current processing state."""
        if self.state == PipelineState.FILTERED and self.filtered is not None:
            return self.filtered
        if self.state == PipelineState.OD and self.od is not None:
            return self.od
        return self.raw

    @property
    def state_label(self) -> str:
        """Human-readable label for the current state."""
        labels = {
            PipelineState.RAW: "Raw Intensity",
            PipelineState.OD: "Optical Density",
            PipelineState.FILTERED: "Filtered OD",
        }
        return labels[self.state]


class ProcessingPipeline:
    """Manages the fNIRS signal processing pipeline.

    Usage
    -----
    >>> pipe = ProcessingPipeline(intensity, sampling_rate)
    >>> pipe.convert_to_od()
    >>> pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    >>> filtered = pipe.result.active_data
    """

    def __init__(self, intensity: np.ndarray, sampling_rate: float):
        self._fs = sampling_rate
        self._result = PipelineResult(
            raw=np.array(intensity, dtype=np.float64),
        )

    @property
    def result(self) -> PipelineResult:
        return self._result

    @property
    def state(self) -> PipelineState:
        return self._result.state

    @property
    def sampling_rate(self) -> float:
        return self._fs

    def convert_to_od(self) -> np.ndarray:
        """Convert raw intensity to optical density.

        Returns the OD array and advances state to OD.
        """
        self._result.od = intensity_to_od(self._result.raw)
        self._result.state = PipelineState.OD
        # Clear any previous filtering (OD changed)
        self._result.filtered = None
        return self._result.od

    def apply_bandpass(
        self,
        low: float = 0.01,
        high: float = 0.1,
        order: int = 3,
    ) -> np.ndarray:
        """Apply bandpass filter to OD data.

        If OD has not been computed yet, it is computed first.

        Returns the filtered array and advances state to FILTERED.
        """
        if self._result.od is None:
            self.convert_to_od()

        self._result.filtered = bandpass_filter(
            self._result.od, self._fs,
            low=low, high=high, order=order,
        )
        self._result.filter_low = low
        self._result.filter_high = high
        self._result.filter_order = order
        self._result.state = PipelineState.FILTERED
        return self._result.filtered

    def set_view(self, state: PipelineState) -> np.ndarray:
        """Switch the active view without recomputing.

        Only allows switching to a state whose data has been computed.
        """
        if state == PipelineState.FILTERED and self._result.filtered is None:
            raise ValueError("Filtered data not available yet")
        if state == PipelineState.OD and self._result.od is None:
            raise ValueError("OD data not available yet")
        self._result.state = state
        return self._result.active_data

    def reset(self):
        """Reset pipeline to raw state, discarding all processed data."""
        self._result.od = None
        self._result.filtered = None
        self._result.filter_low = 0.0
        self._result.filter_high = 0.0
        self._result.filter_order = 0
        self._result.state = PipelineState.RAW
