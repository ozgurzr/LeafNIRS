"""3D brain viewer showing GLM activation on a cortical surface."""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QFrame, QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QLinearGradient, QColor, QPainter, QPixmap, QVector3D

from processing.brain_mesh import load_brain_mesh, project_probes_to_surface
from processing.glm_analysis import GLMResult


class ColorBarWidget(QLabel):
    """Simple horizontal colorbar showing t-statistic gradient."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setMinimumWidth(120)
        self._t_min = -3.0
        self._t_max = 3.0
        self._chrom = "HbO"
        self._update_pixmap()

    def set_range(self, t_min: float, t_max: float, chrom: str = "HbO"):
        self._t_min = t_min
        self._t_max = t_max
        self._chrom = chrom
        self._update_pixmap()

    def _update_pixmap(self):
        w, h = 200, 16
        pm = QPixmap(w, h)
        pm.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pm)

        grad = QLinearGradient(0, 0, w, 0)

        if self._chrom == "HbO":
            grad.setColorAt(0.0, QColor(30, 80, 220, 220))
            grad.setColorAt(0.35, QColor(100, 100, 100, 120))
            grad.setColorAt(0.5, QColor(140, 140, 140, 80))
            grad.setColorAt(0.65, QColor(200, 120, 50, 160))
            grad.setColorAt(1.0, QColor(255, 200, 0, 240))
        else:
            grad.setColorAt(0.0, QColor(0, 200, 255, 240))
            grad.setColorAt(0.35, QColor(60, 120, 200, 160))
            grad.setColorAt(0.5, QColor(140, 140, 140, 80))
            grad.setColorAt(0.65, QColor(200, 120, 50, 160))
            grad.setColorAt(1.0, QColor(255, 180, 0, 240))

        painter.fillRect(0, 0, w, h, grad)
        painter.end()
        self.setPixmap(pm)


class BrainViewerWidget(QWidget):
    """Interactive 3D brain viewer with activation overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._glm_hbo: GLMResult | None = None
        self._glm_hbr: GLMResult | None = None
        self._src_pos_3d: np.ndarray | None = None
        self._det_pos_3d: np.ndarray | None = None
        self._pair_midpoints_3d: np.ndarray | None = None
        self._brain_verts = None
        self._brain_faces = None
        self._label_items: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #2d2d30; border-bottom: 1px solid #3e3e42;")
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel("  3D Brain Map")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #dcdcdc;")

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(self._styled_label("View:"))
        self._combo_chrom = QComboBox()
        self._combo_chrom.addItems(["HbO", "HbR"])
        self._combo_chrom.setFixedWidth(80)
        self._combo_chrom.currentTextChanged.connect(self._update_activation)
        controls.addWidget(self._combo_chrom)

        controls.addWidget(self._styled_label("Condition:"))
        self._combo_cond = QComboBox()
        self._combo_cond.setFixedWidth(100)
        self._combo_cond.currentIndexChanged.connect(self._update_activation)
        controls.addWidget(self._combo_cond)

        controls.addWidget(self._styled_label("p <"))
        self._slider_p = QSlider(Qt.Horizontal)
        self._slider_p.setRange(1, 100)
        self._slider_p.setValue(5)
        self._slider_p.setFixedWidth(100)
        self._slider_p.valueChanged.connect(self._update_activation)
        self._lbl_p = QLabel("0.05")
        self._lbl_p.setFixedWidth(35)
        self._lbl_p.setStyleSheet("color: #dcdcdc;")
        controls.addWidget(self._slider_p)
        controls.addWidget(self._lbl_p)

        self._btn_reset_cam = QPushButton("Reset View")
        self._btn_reset_cam.setFixedHeight(24)
        self._btn_reset_cam.setStyleSheet(
            "QPushButton { background: #3e3e42; color: #dcdcdc; border: none; "
            "border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
            "QPushButton:hover { background: #505054; }"
        )
        self._btn_reset_cam.clicked.connect(self._reset_camera)
        controls.addWidget(self._btn_reset_cam)

        controls.addStretch()

        ctrl_layout.addWidget(title)
        ctrl_layout.addLayout(controls)


        cbar_row = QHBoxLayout()
        cbar_row.setSpacing(4)
        self._lbl_t_min = QLabel("")
        self._lbl_t_min.setStyleSheet("color: #999; font-size: 10px;")
        self._lbl_t_min.setFixedWidth(40)
        self._lbl_t_min.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._colorbar = ColorBarWidget()
        self._lbl_t_max = QLabel("")
        self._lbl_t_max.setStyleSheet("color: #999; font-size: 10px;")
        self._lbl_t_max.setFixedWidth(40)
        self._lbl_cbar_title = QLabel("t-stat")
        self._lbl_cbar_title.setStyleSheet("color: #888; font-size: 9px;")
        self._lbl_cbar_title.setFixedWidth(40)
        cbar_row.addStretch()
        cbar_row.addWidget(self._lbl_cbar_title)
        cbar_row.addWidget(self._lbl_t_min)
        cbar_row.addWidget(self._colorbar)
        cbar_row.addWidget(self._lbl_t_max)
        cbar_row.addStretch()
        ctrl_layout.addLayout(cbar_row)

        layout.addWidget(ctrl_frame)


        self._glview = gl.GLViewWidget()
        self._glview.setBackgroundColor('#1a1a2e')
        self._reset_camera()
        layout.addWidget(self._glview)


        self._lbl_info = QLabel("")
        self._lbl_info.setStyleSheet("color: #888; font-size: 10px; padding: 2px 8px;")
        self._lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_info)


        self._brain_mesh_item = None
        self._src_scatter = None
        self._det_scatter = None
        self._act_scatter = None
        self._line_items = []

        self.setVisible(False)

    @staticmethod
    def _styled_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        return lbl

    def _reset_camera(self):
        self._glview.opts['distance'] = 280
        self._glview.opts['elevation'] = 60
        self._glview.opts['azimuth'] = -90
        self._glview.opts['center'] = QVector3D(0, 0, 10)
        self._glview.update()

    def set_glm_results(self, glm_hbo: GLMResult, glm_hbr: GLMResult,
                        probe, channels):
        """Build 3D visualization from GLM results."""
        self._glm_hbo = glm_hbo
        self._glm_hbr = glm_hbr
        self._clear_items()

        verts, faces = load_brain_mesh('fsaverage5')
        self._brain_verts = verts
        self._brain_faces = faces

        # Z-height gradient simulates ambient occlusion.
        face_center_z = verts[faces].mean(axis=1)[:, 2]
        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        z_norm = (face_center_z - z_min) / (z_max - z_min + 1e-6)
        base = 0.40 + 0.30 * z_norm
        face_colors = np.column_stack([
            base * 0.92,
            base * 0.85,
            base * 0.95,
            np.full(len(faces), 0.50),
        ])

        self._brain_mesh_item = gl.GLMeshItem(
            vertexes=verts, faces=faces,
            faceColors=face_colors,
            smooth=True, drawEdges=False,
            shader='shaded',
        )
        self._brain_mesh_item.setGLOptions('translucent')
        self._glview.addItem(self._brain_mesh_item)

        self._add_anatomical_labels(verts)


        src_2d = probe.source_pos[:, :2]
        det_2d = probe.detector_pos[:, :2]

        self._src_pos_3d = project_probes_to_surface(src_2d, verts, faces)
        self._det_pos_3d = project_probes_to_surface(det_2d, verts, faces)


        midpoints = []
        for lbl in glm_hbo.pair_labels:
            parts = lbl.replace('S', '').replace('D', '').split('-')
            si, di = int(parts[0]) - 1, int(parts[1]) - 1
            mid = (self._src_pos_3d[si] + self._det_pos_3d[di]) / 2.0
            midpoints.append(mid)
        self._pair_midpoints_3d = np.array(midpoints)

        src_colors = np.zeros((len(self._src_pos_3d), 4))
        src_colors[:] = [1.0, 0.3, 0.2, 0.95]
        self._src_scatter = gl.GLScatterPlotItem(
            pos=self._src_pos_3d, size=7, color=src_colors, pxMode=True,
        )
        self._glview.addItem(self._src_scatter)

        det_colors = np.zeros((len(self._det_pos_3d), 4))
        det_colors[:] = [0.2, 0.5, 1.0, 0.95]
        self._det_scatter = gl.GLScatterPlotItem(
            pos=self._det_pos_3d, size=5, color=det_colors, pxMode=True,
        )
        self._glview.addItem(self._det_scatter)

        for lbl in glm_hbo.pair_labels:
            parts = lbl.replace('S', '').replace('D', '').split('-')
            si, di = int(parts[0]) - 1, int(parts[1]) - 1
            line = gl.GLLinePlotItem(
                pos=np.array([self._src_pos_3d[si], self._det_pos_3d[di]]),
                color=(0.5, 0.5, 0.55, 0.25), width=1.5, antialias=True,
            )
            self._glview.addItem(line)
            self._line_items.append(line)

        self._act_scatter = gl.GLScatterPlotItem(
            pos=self._pair_midpoints_3d, size=12, pxMode=True,
        )
        self._glview.addItem(self._act_scatter)


        self._combo_cond.blockSignals(True)
        self._combo_cond.clear()
        for name in glm_hbo.contrast_names:
            self._combo_cond.addItem(name)
        self._combo_cond.blockSignals(False)

        self.setVisible(True)
        self._reset_camera()
        self._update_activation()

    def _add_anatomical_labels(self, verts: np.ndarray):
        """Add L/R/A/P text labels around the brain."""
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        z_top = verts[:, 2].max()

        label_defs = [
            ("L", [x_min - 25, 0, z_top * 0.7], (1.0, 0.5, 0.5, 0.8)),
            ("R", [x_max + 25, 0, z_top * 0.7], (1.0, 0.5, 0.5, 0.8)),
            ("A", [0, y_max + 25, z_top * 0.7], (0.5, 0.85, 1.0, 0.8)),
            ("P", [0, y_min - 25, z_top * 0.7], (0.5, 0.85, 1.0, 0.8)),
        ]

        for text, pos, color in label_defs:
            item = gl.GLTextItem(
                pos=np.array(pos, dtype=np.float32),
                text=text,
                color=color,
                font=QFont("Segoe UI", 14, QFont.Bold),
            )
            self._glview.addItem(item)
            self._label_items.append(item)

    def _update_activation(self):
        if self._glm_hbo is None or self._pair_midpoints_3d is None:
            return

        chrom = self._combo_chrom.currentText()
        cond_idx = self._combo_cond.currentIndex()
        p_thresh = self._slider_p.value() / 100.0
        self._lbl_p.setText(f"{p_thresh:.2f}")

        if cond_idx < 0:
            return

        glm = self._glm_hbo if chrom == "HbO" else self._glm_hbr
        t_vals = glm.t_stat[cond_idx, :]
        p_vals = glm.p_value[cond_idx, :]

        t_abs_max = max(np.abs(t_vals).max(), 1e-6)
        t_norm = t_vals / t_abs_max

        colors = np.zeros((len(t_vals), 4))
        sizes = np.full(len(t_vals), 4.0)

        n_sig = 0
        for i in range(len(t_vals)):
            significant = p_vals[i] < p_thresh
            intensity = min(abs(t_norm[i]), 1.0)

            if significant:
                n_sig += 1
                sizes[i] = 10.0 + intensity * 14.0
                if chrom == "HbO":
                    if t_vals[i] >= 0:
                        colors[i] = [1.0, 0.4 + 0.6 * intensity, 0.0, 0.92]
                    else:
                        colors[i] = [0.1, 0.3 + 0.5 * intensity, 1.0, 0.92]
                else:
                    if t_vals[i] <= 0:
                        colors[i] = [0.0, 0.5 + 0.5 * intensity, 1.0, 0.92]
                    else:
                        colors[i] = [1.0, 0.4 + 0.5 * intensity, 0.0, 0.92]
            else:
                sizes[i] = 4.0
                colors[i] = [0.4, 0.4, 0.4, 0.15]

        self._act_scatter.setData(
            pos=self._pair_midpoints_3d,
            size=sizes,
            color=colors,
        )

        self._colorbar.set_range(-t_abs_max, t_abs_max, chrom)
        self._lbl_t_min.setText(f"{-t_abs_max:.1f}")
        self._lbl_t_max.setText(f"{t_abs_max:.1f}")

        cond_name = self._combo_cond.currentText()
        self._lbl_info.setText(
            f"{chrom}  |  Condition '{cond_name}'  |  "
            f"{n_sig}/{len(t_vals)} significant (p<{p_thresh:.2f})  |  "
            f"t-range: [{t_vals.min():.1f}, {t_vals.max():.1f}]"
        )

    def _clear_items(self):
        for item in [self._brain_mesh_item, self._src_scatter,
                     self._det_scatter, self._act_scatter]:
            if item is not None:
                self._glview.removeItem(item)
        for line in self._line_items:
            self._glview.removeItem(line)
        for lbl in self._label_items:
            self._glview.removeItem(lbl)
        self._line_items = []
        self._label_items = []
        self._brain_mesh_item = None
        self._src_scatter = None
        self._det_scatter = None
        self._act_scatter = None

    def clear(self):
        self._clear_items()
        self._glm_hbo = None
        self._glm_hbr = None
        self._pair_midpoints_3d = None
        self._lbl_info.setText("")
        self._lbl_t_min.setText("")
        self._lbl_t_max.setText("")
        self.setVisible(False)
