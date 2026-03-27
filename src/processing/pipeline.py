"""
Processing Pipeline Manager.

Orchestrates the signal processing stages and tracks the current
processing state. Stores intermediate results so the user can
switch between views (Raw / OD / Filtered / HbO-HbR).
"""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np

from processing.od_converter import intensity_to_od
from processing.bandpass_filter import bandpass_filter
from processing.mbll_converter import od_to_concentration

from data_io.snirf_loader_base import ChannelInfo, ProbeGeometry


class PipelineState(Enum):
    """Current stage of the processing pipeline."""
    RAW = auto()
    OD = auto()
    FILTERED = auto()
    CONCENTRATION = auto()


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    raw: np.ndarray                         # (n_time, n_ch) original intensity
    od: np.ndarray | None = None            # (n_time, n_ch) optical density
    filtered: np.ndarray | None = None      # (n_time, n_ch) bandpass filtered OD
    hbo: np.ndarray | None = None           # (n_time, n_pairs) ΔHbO
    hbr: np.ndarray | None = None           # (n_time, n_pairs) ΔHbR
    pair_labels: list[str] = field(default_factory=list)
    state: PipelineState = PipelineState.RAW

    # Filter parameters used (for display)
    filter_low: float = 0.0
    filter_high: float = 0.0
    filter_order: int = 0

    @property
    def active_data(self) -> np.ndarray:
        """Return the data array for the current processing state."""
        if self.state == PipelineState.CONCENTRATION and self.hbo is not None:
            return self.hbo  # default: show HbO
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
            PipelineState.CONCENTRATION: "HbO / HbR",
        }
        return labels[self.state]


class ProcessingPipeline:
    """Manages the fNIRS signal processing pipeline.

    Usage
    -----
    >>> pipe = ProcessingPipeline(intensity, sampling_rate, channels=ch, probe=pr)
    >>> pipe.convert_to_od()
    >>> pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    >>> pipe.convert_to_concentration()
    >>> hbo, hbr = pipe.result.hbo, pipe.result.hbr
    """

    def __init__(
        self,
        intensity: np.ndarray,
        sampling_rate: float,
        channels: list[ChannelInfo] | None = None,
        probe: ProbeGeometry | None = None,
    ):
        self._fs = sampling_rate
        self._channels = channels
        self._probe = probe
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
        # Clear downstream results
        self._result.filtered = None
        self._result.hbo = None
        self._result.hbr = None
        self._result.pair_labels = []
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
        # Clear downstream
        self._result.hbo = None
        self._result.hbr = None
        self._result.pair_labels = []
        return self._result.filtered

    def convert_to_concentration(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert OD/filtered data to HbO/HbR concentration via MBLL.

        Uses filtered OD if available, otherwise raw OD.
        If neither is computed, the full pipeline is run first.

        Returns (hbo, hbr) arrays and advances state to CONCENTRATION.
        """
        if self._channels is None or self._probe is None:
            raise ValueError(
                "Channel and probe info required for MBLL. "
                "Pass channels= and probe= to ProcessingPipeline()."
            )

        # Use best available OD data
        if self._result.filtered is not None:
            source_od = self._result.filtered
        elif self._result.od is not None:
            source_od = self._result.od
        else:
            self.convert_to_od()
            source_od = self._result.od

        hbo, hbr, labels = od_to_concentration(
            source_od, self._channels, self._probe,
        )
        self._result.hbo = hbo
        self._result.hbr = hbr
        self._result.pair_labels = labels
        self._result.state = PipelineState.CONCENTRATION
        return hbo, hbr

    def set_view(self, state: PipelineState) -> np.ndarray:
        """Switch the active view without recomputing.

        Only allows switching to a state whose data has been computed.
        """
        if state == PipelineState.CONCENTRATION and self._result.hbo is None:
            raise ValueError("Concentration data not available yet")
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
        self._result.hbo = None
        self._result.hbr = None
        self._result.pair_labels = []
        self._result.filter_low = 0.0
        self._result.filter_high = 0.0
        self._result.filter_order = 0
        self._result.state = PipelineState.RAW
