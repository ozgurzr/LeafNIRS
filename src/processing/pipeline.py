"""Processing pipeline: tracks state and stores intermediate results.

Pipeline: Raw → OD → Corrected → Filtered → HbO/HbR
"""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np

from processing.od_converter import intensity_to_od
from processing.bandpass_filter import bandpass_filter
from processing.mbll_converter import od_to_concentration
from processing.motion_correction import detect_artifacts, correct_tddr, correct_spline
from data_io.snirf_loader_base import ChannelInfo, ProbeGeometry


class PipelineState(Enum):
    RAW = auto()
    OD = auto()
    CORRECTED = auto()
    FILTERED = auto()
    CONCENTRATION = auto()


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    raw: np.ndarray
    od: np.ndarray | None = None
    corrected: np.ndarray | None = None
    artifact_mask: np.ndarray | None = None
    filtered: np.ndarray | None = None
    hbo: np.ndarray | None = None
    hbr: np.ndarray | None = None
    pair_labels: list[str] = field(default_factory=list)
    state: PipelineState = PipelineState.RAW
    filter_low: float = 0.0
    filter_high: float = 0.0
    filter_order: int = 0
    correction_method: str = ""

    @property
    def active_data(self) -> np.ndarray:
        """Return the data array for the current processing state."""
        if self.state == PipelineState.CONCENTRATION and self.hbo is not None:
            return self.hbo
        if self.state == PipelineState.FILTERED and self.filtered is not None:
            return self.filtered
        if self.state == PipelineState.CORRECTED and self.corrected is not None:
            return self.corrected
        if self.state == PipelineState.OD and self.od is not None:
            return self.od
        return self.raw

    @property
    def state_label(self) -> str:
        return {
            PipelineState.RAW: "Raw Intensity",
            PipelineState.OD: "Optical Density",
            PipelineState.CORRECTED: "Motion Corrected",
            PipelineState.FILTERED: "Filtered OD",
            PipelineState.CONCENTRATION: "HbO / HbR",
        }[self.state]


class ProcessingPipeline:
    """Manages the fNIRS signal processing pipeline.

    >>> pipe = ProcessingPipeline(intensity, fs, channels=ch, probe=pr)
    >>> pipe.convert_to_od()
    >>> pipe.apply_motion_correction(method='tddr')
    >>> pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    >>> pipe.convert_to_concentration()
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
        self._result = PipelineResult(raw=np.array(intensity, dtype=np.float64))

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
        """Convert raw intensity to optical density."""
        self._result.od = intensity_to_od(self._result.raw)
        self._result.state = PipelineState.OD
        self._clear_downstream('od')
        return self._result.od

    def apply_motion_correction(self, method: str = 'tddr') -> np.ndarray:
        """Apply motion artifact correction ('tddr' or 'spline')."""
        if self._result.od is None:
            self.convert_to_od()

        od = self._result.od
        self._result.artifact_mask = detect_artifacts(od, self._fs)

        if method == 'spline':
            self._result.corrected = correct_spline(od, self._result.artifact_mask)
        else:
            self._result.corrected = correct_tddr(od)

        self._result.correction_method = method
        self._result.state = PipelineState.CORRECTED
        self._clear_downstream('corrected')
        return self._result.corrected

    def apply_bandpass(
        self,
        low: float = 0.01,
        high: float = 0.1,
        order: int = 3,
    ) -> np.ndarray:
        """Apply bandpass filter to best available OD data."""
        if self._result.corrected is not None:
            source = self._result.corrected
        elif self._result.od is not None:
            source = self._result.od
        else:
            self.convert_to_od()
            source = self._result.od

        self._result.filtered = bandpass_filter(source, self._fs, low=low, high=high, order=order)
        self._result.filter_low = low
        self._result.filter_high = high
        self._result.filter_order = order
        self._result.state = PipelineState.FILTERED
        self._clear_downstream('filtered')
        return self._result.filtered

    def convert_to_concentration(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert OD/filtered data to HbO/HbR via MBLL."""
        if self._channels is None or self._probe is None:
            raise ValueError(
                "Channel and probe info required for MBLL. "
                "Pass channels= and probe= to ProcessingPipeline()."
            )

        if self._result.filtered is not None:
            source_od = self._result.filtered
        elif self._result.corrected is not None:
            source_od = self._result.corrected
        elif self._result.od is not None:
            source_od = self._result.od
        else:
            self.convert_to_od()
            source_od = self._result.od

        hbo, hbr, labels = od_to_concentration(source_od, self._channels, self._probe)
        self._result.hbo = hbo
        self._result.hbr = hbr
        self._result.pair_labels = labels
        self._result.state = PipelineState.CONCENTRATION
        return hbo, hbr

    def set_view(self, state: PipelineState) -> np.ndarray:
        """Switch the active view without recomputing."""
        if state == PipelineState.CONCENTRATION and self._result.hbo is None:
            raise ValueError("Concentration data not available yet")
        if state == PipelineState.FILTERED and self._result.filtered is None:
            raise ValueError("Filtered data not available yet")
        if state == PipelineState.CORRECTED and self._result.corrected is None:
            raise ValueError("Corrected data not available yet")
        if state == PipelineState.OD and self._result.od is None:
            raise ValueError("OD data not available yet")
        self._result.state = state
        return self._result.active_data

    def reset(self):
        """Reset pipeline to raw state."""
        self._result.od = None
        self._result.corrected = None
        self._result.artifact_mask = None
        self._result.filtered = None
        self._result.hbo = None
        self._result.hbr = None
        self._result.pair_labels = []
        self._result.filter_low = 0.0
        self._result.filter_high = 0.0
        self._result.filter_order = 0
        self._result.correction_method = ""
        self._result.state = PipelineState.RAW

    def _clear_downstream(self, from_stage: str):
        """Clear all stages after the given stage."""
        stages = ['od', 'corrected', 'filtered', 'concentration']
        idx = stages.index(from_stage) if from_stage in stages else -1
        if idx < len(stages) - 1:
            if from_stage in ('od', 'corrected'):
                self._result.filtered = None
            if from_stage != 'concentration':
                self._result.hbo = None
                self._result.hbr = None
                self._result.pair_labels = []
        if from_stage == 'od':
            self._result.corrected = None
            self._result.artifact_mask = None
