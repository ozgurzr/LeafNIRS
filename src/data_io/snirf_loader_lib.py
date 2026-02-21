"""SNIRF Loader — Method A: snirf library wrapper."""
from __future__ import annotations

import numpy as np
from snirf import Snirf

from data_io.snirf_loader_base import (
    SNIRFLoaderBase, SNIRFData, ProbeGeometry,
    ChannelInfo, StimulusInfo,
)


class SNIRFLoaderLib(SNIRFLoaderBase):
    """Loads SNIRF files using the ``snirf`` Python library."""

    @staticmethod
    def loader_name() -> str:
        return "snirf-library"

    def _load_impl(self, filepath: str) -> SNIRFData:
        with Snirf(filepath, 'r') as s:
            nirs = s.nirs[0]

            intensity = np.array(nirs.data[0].dataTimeSeries, dtype=np.float64)
            time = np.array(nirs.data[0].time, dtype=np.float64).ravel()

            probe = nirs.probe
            if hasattr(probe, 'sourcePos3D') and probe.sourcePos3D is not None:
                src_pos = np.array(probe.sourcePos3D, dtype=np.float64)
            else:
                src_pos = np.array(probe.sourcePos2D, dtype=np.float64)
            if hasattr(probe, 'detectorPos3D') and probe.detectorPos3D is not None:
                det_pos = np.array(probe.detectorPos3D, dtype=np.float64)
            else:
                det_pos = np.array(probe.detectorPos2D, dtype=np.float64)

            wavelengths = np.array(probe.wavelengths, dtype=np.float64).ravel()

            src_labels = [f"S{i+1}" for i in range(src_pos.shape[0])]
            det_labels = [f"D{i+1}" for i in range(det_pos.shape[0])]

            probe_geom = ProbeGeometry(
                source_pos=src_pos, detector_pos=det_pos,
                wavelengths=wavelengths,
                source_labels=src_labels, detector_labels=det_labels,
            )

            channels = []
            for ml in nirs.data[0].measurementList:
                channels.append(ChannelInfo(
                    source_index=int(ml.sourceIndex),
                    detector_index=int(ml.detectorIndex),
                    wavelength_index=int(ml.wavelengthIndex),
                    data_type=int(ml.dataType) if ml.dataType else 0,
                    data_type_label=str(ml.dataTypeLabel) if ml.dataTypeLabel else "",
                ))

            stimuli = []
            if nirs.stim:
                for stim in nirs.stim:
                    if stim.data is not None and len(stim.data) > 0:
                        sd = np.array(stim.data, dtype=np.float64)
                        if sd.ndim == 2 and sd.shape[0] > 0:
                            stimuli.append(StimulusInfo(
                                name=str(stim.name) if stim.name else "unnamed",
                                onset=sd[:, 0],
                                duration=sd[:, 1] if sd.shape[1] > 1 else np.ones(len(sd)),
                                amplitude=sd[:, 2] if sd.shape[1] > 2 else np.ones(len(sd)),
                            ))

            metadata = {}
            if nirs.metaDataTags:
                for attr in dir(nirs.metaDataTags):
                    if attr.startswith('_'):
                        continue
                    try:
                        val = getattr(nirs.metaDataTags, attr)
                        if val is not None and not callable(val):
                            metadata[attr] = str(val) if not isinstance(val, (int, float)) else val
                    except Exception:
                        pass

        return SNIRFData(
            intensity=intensity, time=time, probe=probe_geom,
            channels=channels, stimuli=stimuli, metadata=metadata,
        )
