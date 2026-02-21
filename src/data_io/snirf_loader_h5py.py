"""SNIRF Loader — Method B: raw h5py HDF5 access."""
from __future__ import annotations

import h5py
import numpy as np

from data_io.snirf_loader_base import (
    SNIRFLoaderBase, SNIRFData, ProbeGeometry,
    ChannelInfo, StimulusInfo,
)


def _unwrap_scalar(val):
    """Convert numpy (1,)-shaped arrays and bytes to plain Python types."""
    if isinstance(val, np.ndarray):
        if val.dtype.kind == 'S':
            decoded = [v.decode('utf-8') for v in val.ravel()]
            return decoded[0] if val.size == 1 else decoded
        if val.size == 1:
            return val.item()
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return val


def _read_dataset(group, name, default=None):
    """Safely read an HDF5 dataset, returning default if absent."""
    if name in group:
        return _unwrap_scalar(group[name][()])
    return default


class SNIRFLoaderH5py(SNIRFLoaderBase):
    """Loads SNIRF files using raw h5py for direct HDF5 traversal."""

    @staticmethod
    def loader_name() -> str:
        return "h5py-raw"

    def _load_impl(self, filepath: str) -> SNIRFData:
        with h5py.File(filepath, 'r') as f:
            nirs_key = 'nirs' if 'nirs' in f else 'nirs1'
            if nirs_key not in f:
                raise ValueError("No nirs group found in SNIRF file")
            nirs = f[nirs_key]

            data_key = 'data' if 'data' in nirs else 'data1'
            data_grp = nirs[data_key]
            intensity = np.array(data_grp['dataTimeSeries'], dtype=np.float64)
            time = np.array(data_grp['time'], dtype=np.float64).ravel()

            probe_grp = nirs['probe']
            src_pos = np.array(probe_grp.get('sourcePos3D', probe_grp.get('sourcePos2D')), dtype=np.float64)
            det_pos = np.array(probe_grp.get('detectorPos3D', probe_grp.get('detectorPos2D')), dtype=np.float64)
            wavelengths = np.array(probe_grp['wavelengths'], dtype=np.float64).ravel()

            src_labels = _read_dataset(probe_grp, 'sourceLabels')
            src_labels = [f"S{i+1}" for i in range(src_pos.shape[0])] if src_labels is None else (list(src_labels) if not isinstance(src_labels, list) else src_labels)

            det_labels = _read_dataset(probe_grp, 'detectorLabels')
            det_labels = [f"D{i+1}" for i in range(det_pos.shape[0])] if det_labels is None else (list(det_labels) if not isinstance(det_labels, list) else det_labels)

            probe_geom = ProbeGeometry(
                source_pos=src_pos, detector_pos=det_pos,
                wavelengths=wavelengths,
                source_labels=src_labels, detector_labels=det_labels,
            )

            # Parse measurement list
            channels: list[ChannelInfo] = []
            ml_idx = 1
            while True:
                ml_key = f'measurementList{ml_idx}'
                if ml_key not in data_grp:
                    break
                ml_grp = data_grp[ml_key]
                channels.append(ChannelInfo(
                    source_index=int(_read_dataset(ml_grp, 'sourceIndex', 0)),
                    detector_index=int(_read_dataset(ml_grp, 'detectorIndex', 0)),
                    wavelength_index=int(_read_dataset(ml_grp, 'wavelengthIndex', 0)),
                    data_type=int(_read_dataset(ml_grp, 'dataType', 0)),
                    data_type_label=str(_read_dataset(ml_grp, 'dataTypeLabel', '')),
                ))
                ml_idx += 1

            # Parse stimuli
            stimuli: list[StimulusInfo] = []
            stim_idx = 1
            while True:
                stim_key = f'stim{stim_idx}'
                if stim_key not in nirs:
                    break
                stim_grp = nirs[stim_key]
                name = _read_dataset(stim_grp, 'name', f'stim{stim_idx}')
                stim_data_raw = _read_dataset(stim_grp, 'data')
                if stim_data_raw is not None:
                    stim_arr = np.array(stim_data_raw, dtype=np.float64)
                    if stim_arr.ndim == 2 and stim_arr.shape[0] > 0:
                        stimuli.append(StimulusInfo(
                            name=str(name),
                            onset=stim_arr[:, 0],
                            duration=stim_arr[:, 1] if stim_arr.shape[1] > 1 else np.ones(len(stim_arr)),
                            amplitude=stim_arr[:, 2] if stim_arr.shape[1] > 2 else np.ones(len(stim_arr)),
                        ))
                stim_idx += 1

            metadata: dict = {}
            if 'metaDataTags' in nirs:
                mdt = nirs['metaDataTags']
                for key in mdt.keys():
                    val = _read_dataset(mdt, key)
                    if val is not None:
                        metadata[key] = val if not isinstance(val, np.ndarray) else val.tolist()

        return SNIRFData(
            intensity=intensity, time=time, probe=probe_geom,
            channels=channels, stimuli=stimuli, metadata=metadata,
        )
