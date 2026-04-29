"""Full end-to-end test of LeafNIRS pipeline including GLM and 3D brain."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data_io.snirf_loader_h5py import SNIRFLoaderH5py
from processing.pipeline import ProcessingPipeline
from processing.epoch_extraction import compute_condition_average
from processing.brain_mesh import generate_brain_mesh, project_2d_to_brain

SNIRF = r"C:\Users\90546\Desktop\LeafNIRS Plan\data\DATA_MotorNeuron\SUBJID_6082\run1.snirf"


def main():
    print("=== 1. Loading SNIRF ===")
    data = SNIRFLoaderH5py().load(SNIRF)
    print(f"  {data.n_channels} ch, {data.n_timepoints} tp, {data.sampling_rate:.1f} Hz")
    print(f"  Stimuli: {[s.name for s in data.stimuli]}")

    print("\n=== 2. Full Pipeline ===")
    pipe = ProcessingPipeline(data.intensity, data.sampling_rate,
                              channels=data.channels, probe=data.probe)
    pipe.apply_motion_correction(method='tddr')
    pipe.apply_bandpass(low=0.01, high=0.1, order=3)
    pipe.convert_to_concentration()
    print(f"  HbO: {pipe.result.hbo.shape}, HbR: {pipe.result.hbr.shape}")
    print(f"  Pairs: {len(pipe.result.pair_labels)}")

    print("\n=== 3. GLM Analysis ===")
    glm_hbo, glm_hbr = pipe.run_glm(data.stimuli)
    print(f"  Conditions: {glm_hbo.contrast_names}")
    print(f"  Beta: {glm_hbo.beta.shape}, T-stat: {glm_hbo.t_stat.shape}")
    n_sig = int(np.sum(glm_hbo.p_value < 0.05))
    total = glm_hbo.beta.shape[0] * glm_hbo.beta.shape[1]
    print(f"  Significant HbO (p<0.05): {n_sig}/{total}")
    for ci, name in enumerate(glm_hbo.contrast_names):
        sig = int(np.sum(glm_hbo.p_value[ci] < 0.05))
        max_t = float(np.max(np.abs(glm_hbo.t_stat[ci])))
        print(f"    Cond '{name}': {sig}/60 sig, max|t|={max_t:.2f}")

    print("\n=== 4. Brain Mesh ===")
    verts, faces = generate_brain_mesh(resolution=50)
    print(f"  Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}")
    print(f"  X: [{verts[:,0].min():.0f}, {verts[:,0].max():.0f}]")
    print(f"  Y: [{verts[:,1].min():.0f}, {verts[:,1].max():.0f}]")
    print(f"  Z: [{verts[:,2].min():.0f}, {verts[:,2].max():.0f}]")

    print("\n=== 5. Probe 2D->3D Projection ===")
    src_3d = project_2d_to_brain(data.probe.source_pos[:, :2], verts)
    det_3d = project_2d_to_brain(data.probe.detector_pos[:, :2], verts)
    print(f"  Sources: {src_3d.shape}, Z=[{src_3d[:,2].min():.1f}, {src_3d[:,2].max():.1f}]")
    print(f"  Detectors: {det_3d.shape}, Z=[{det_3d[:,2].min():.1f}, {det_3d[:,2].max():.1f}]")

    print("\n=== 6. Block Average ===")
    stim = data.stimuli[0]
    ba = compute_condition_average(
        hbo=pipe.result.hbo, hbr=pipe.result.hbr,
        time=data.time, onsets=stim.onset,
        condition_name=stim.name, pair_labels=pipe.result.pair_labels,
        pre_sec=2.0, post_sec=20.0,
    )
    print(f"  Condition: {ba.condition}, Trials: {ba.n_trials}")
    print(f"  HbO mean: {ba.hbo_mean.shape}")

    print("\n=== 7. GUI Imports ===")
    from gui.main_window import MainWindow
    from gui.probe_map_widget import ProbeMapWidget
    from gui.brain_viewer_widget import BrainViewerWidget
    print("  All GUI modules OK")

    print("\n" + "=" * 50)
    print("  ALL 7 CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
