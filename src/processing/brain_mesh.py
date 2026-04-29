"""Brain surface mesh loading and optode-to-surface projection.

Backends:
  - nilearn fsaverage (preferred): anatomically accurate FreeSurfer pial surface.
  - Parametric ellipsoid (fallback): procedural mesh when nilearn is unavailable.
"""
from __future__ import annotations

import logging
import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def load_brain_mesh(resolution: str = 'fsaverage5') -> tuple[np.ndarray, np.ndarray]:
    """Load a combined-hemisphere brain surface mesh.

    Returns
    -------
    vertices : (N, 3) float32, coordinates in mm (MNI305 space for fsaverage).
    faces : (M, 3) int32, triangle indices.
    """
    try:
        verts, faces = _load_fsaverage(resolution)
        log.info("Loaded %s cortical mesh: %d vertices, %d faces",
                 resolution, len(verts), len(faces))
        return verts, faces
    except Exception as exc:
        log.warning("fsaverage unavailable (%s); falling back to parametric mesh", exc)
        return _generate_parametric_mesh(resolution=50)


def project_probes_to_surface(
    pos_2d: np.ndarray,
    brain_verts: np.ndarray,
    brain_faces: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Map 2D probe layout onto the dorsal cortex using KD-tree nearest-vertex lookup.

    Parameters
    ----------
    pos_2d : (N, 2+) probe positions in 2D layout space.
    brain_verts : (V, 3) brain mesh vertex coordinates.
    brain_faces : (F, 3) face indices (reserved for future normal-based offset).
    scale : coverage multiplier (1.0 ≈ 35% of dorsal extent).

    Returns
    -------
    pos_3d : (N, 3) projected positions slightly above the cortical surface.
    """
    if pos_2d.shape[0] == 0 or pos_2d.shape[1] < 2:
        return pos_2d

    # Restrict to dorsal (upper 50% by Z) vertices.
    z_median = np.median(brain_verts[:, 2])
    upper_mask = brain_verts[:, 2] > z_median
    upper_verts = brain_verts[upper_mask]
    if len(upper_verts) < 10:
        upper_verts = brain_verts

    # Normalize probe positions to [-1, 1].
    center_2d = pos_2d[:, :2].mean(axis=0)
    centered = pos_2d[:, :2] - center_2d
    extent = max(
        centered[:, 0].max() - centered[:, 0].min(),
        centered[:, 1].max() - centered[:, 1].min(),
        1.0,
    )
    normalized = centered / (extent / 2.0)

    # Scale to dorsal surface extent.
    upper_x_range = upper_verts[:, 0].max() - upper_verts[:, 0].min()
    upper_y_range = upper_verts[:, 1].max() - upper_verts[:, 1].min()
    coverage = 0.35 * scale

    target_x = normalized[:, 0] * upper_x_range * coverage + upper_verts[:, 0].mean()
    target_y = normalized[:, 1] * upper_y_range * coverage + upper_verts[:, 1].mean()

    tree_2d = cKDTree(upper_verts[:, :2])
    search_r = max(upper_x_range * 0.06, 10.0)

    pos_3d = np.zeros((len(pos_2d), 3))
    for i in range(len(pos_2d)):
        query_pt = [target_x[i], target_y[i]]
        nearby_idx = tree_2d.query_ball_point(query_pt, r=search_r)

        if nearby_idx:
            best = nearby_idx[np.argmax(upper_verts[nearby_idx, 2])]
            pos_3d[i] = upper_verts[best].copy()
        else:
            _, nearest = tree_2d.query(query_pt)
            pos_3d[i] = upper_verts[nearest].copy()

        pos_3d[i, 2] += 2.0  # offset above surface to avoid z-fighting

    return pos_3d


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-vertex normals for smooth shading."""
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return normals / norms


def _load_fsaverage(resolution: str = 'fsaverage5') -> tuple[np.ndarray, np.ndarray]:
    """Fetch FreeSurfer fsaverage pial surface via nilearn and merge hemispheres."""
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage(mesh=resolution)
    mesh_l = surface.load_surf_mesh(fs['pial_left'])
    mesh_r = surface.load_surf_mesh(fs['pial_right'])

    verts_l = np.asarray(mesh_l.coordinates, dtype=np.float32)
    faces_l = np.asarray(mesh_l.faces, dtype=np.int32)
    verts_r = np.asarray(mesh_r.coordinates, dtype=np.float32)
    faces_r = np.asarray(mesh_r.faces, dtype=np.int32)

    # Right-hemisphere face indices are offset by the left vertex count.
    verts = np.vstack([verts_l, verts_r])
    faces = np.vstack([faces_l, faces_r + len(verts_l)])
    return verts, faces


def _generate_parametric_mesh(resolution: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Generate a procedural cortical-shaped mesh (ellipsoid + Gaussian bumps).

    Serves as fallback when nilearn is not installed.
    """
    u = np.linspace(0, np.pi, resolution)
    v = np.linspace(0, 2 * np.pi, resolution * 2)
    u, v = np.meshgrid(u, v)

    a, b, c = 90.0, 110.0, 65.0
    x = a * np.sin(u) * np.cos(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(u)

    rng = np.random.RandomState(42)
    for _ in range(25):
        cx_ = rng.uniform(-0.9, 0.9)
        cy_ = rng.uniform(-0.9, 0.9)
        cz_ = rng.uniform(-0.2, 0.9)
        amp = rng.uniform(2.5, 6.0)
        width = rng.uniform(0.25, 0.55)
        dx = np.sin(u) * np.cos(v) - cx_
        dy = np.sin(u) * np.sin(v) - cy_
        dz = np.cos(u) - cz_
        bump = amp * np.exp(-(dx**2 + dy**2 + dz**2) / (2 * width**2))
        x += bump * np.sin(u) * np.cos(v)
        y += bump * np.sin(u) * np.sin(v)
        z += bump * np.cos(u)

    # Flatten inferior surface.
    mask = z < -40
    z[mask] = -40 + (z[mask] + 40) * 0.3

    # Longitudinal fissure indentation.
    z -= 12.0 * np.exp(-x**2 / 150.0) * np.clip(z / c, 0, 1)

    rows, cols = u.shape
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces.append([idx, idx + 1, idx + cols])
            faces.append([idx + 1, idx + cols + 1, idx + cols])

    return verts, np.array(faces, dtype=np.int32)


# Backward-compatible aliases.
generate_brain_mesh = _generate_parametric_mesh
project_2d_to_brain = project_probes_to_surface
