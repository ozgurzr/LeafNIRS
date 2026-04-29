"""2D probe activation map showing GLM t-statistics on the probe geometry."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QFrame, QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from processing.glm_analysis import GLMResult


class ProbeMapWidget(QWidget):
    """Displays GLM activation on 2D probe geometry."""

    COLORMAPS = {
        'HbO': pg.colormap.get('CET-L4'),   # red-yellow
        'HbR': pg.colormap.get('CET-L6'),   # blue-cyan
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._probe = None
        self._glm_hbo: GLMResult | None = None
        self._glm_hbr: GLMResult | None = None
        self._pair_midpoints: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("  Probe Activation Map")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #dcdcdc;")

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Chromophore:"))
        self._combo_chrom = QComboBox()
        self._combo_chrom.addItems(["HbO", "HbR"])
        self._combo_chrom.setFixedWidth(80)
        self._combo_chrom.currentTextChanged.connect(self._update_map)
        controls.addWidget(self._combo_chrom)

        controls.addWidget(QLabel("Condition:"))
        self._combo_cond = QComboBox()
        self._combo_cond.setFixedWidth(120)
        self._combo_cond.currentIndexChanged.connect(self._update_map)
        controls.addWidget(self._combo_cond)

        controls.addWidget(QLabel("p <"))
        self._slider_p = QSlider(Qt.Horizontal)
        self._slider_p.setRange(1, 100)
        self._slider_p.setValue(5)
        self._slider_p.setFixedWidth(120)
        self._slider_p.valueChanged.connect(self._update_map)
        self._lbl_p = QLabel("0.05")
        self._lbl_p.setFixedWidth(40)
        controls.addWidget(self._slider_p)
        controls.addWidget(self._lbl_p)

        controls.addStretch()

        ctrl_frame = QFrame()
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)

        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch()
        ctrl_layout.addLayout(title_row)
        ctrl_layout.addLayout(controls)

        ctrl_frame.setStyleSheet("background-color: #2d2d30; border-bottom: 1px solid #3e3e42;")
        layout.addWidget(ctrl_frame)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setLabel('bottom', 'X', units='mm')
        self._plot.setLabel('left', 'Y', units='mm')
        self._plot.showGrid(x=True, y=True, alpha=0.1)
        self._plot.setBackground('#1e1e1e')
        layout.addWidget(self._plot)

        self._scatter = pg.ScatterPlotItem()
        self._plot.addItem(self._scatter)


        self._src_scatter = pg.ScatterPlotItem()
        self._det_scatter = pg.ScatterPlotItem()
        self._plot.addItem(self._src_scatter)
        self._plot.addItem(self._det_scatter)

        self._colorbar = None
        self.setVisible(False)

    def set_glm_results(self, glm_hbo: GLMResult, glm_hbr: GLMResult,
                        probe, channels):
        """Load GLM results and probe geometry for visualization."""
        self._glm_hbo = glm_hbo
        self._glm_hbr = glm_hbr
        self._probe = probe


        src_pos = probe.source_pos[:, :2]
        det_pos = probe.detector_pos[:, :2]


        midpoints = []
        for lbl in glm_hbo.pair_labels:
            parts = lbl.replace('S', '').replace('D', '').split('-')
            si = int(parts[0]) - 1
            di = int(parts[1]) - 1
            mid = (src_pos[si] + det_pos[di]) / 2.0
            midpoints.append(mid)
        self._pair_midpoints = np.array(midpoints)


        self._src_scatter.setData(
            pos=src_pos, size=14, symbol='o',
            pen=pg.mkPen('#ff4444', width=2),
            brush=pg.mkBrush('#ff444480'),
        )
        self._det_scatter.setData(
            pos=det_pos, size=14, symbol='s',
            pen=pg.mkPen('#4488ff', width=2),
            brush=pg.mkBrush('#4488ff80'),
        )


        for item in getattr(self, '_label_items', []):
            self._plot.removeItem(item)
        self._label_items = []

        src_labels = probe.source_labels if probe.source_labels else [f"S{i+1}" for i in range(len(src_pos))]
        det_labels = probe.detector_labels if probe.detector_labels else [f"D{i+1}" for i in range(len(det_pos))]

        for i, pos in enumerate(src_pos):
            txt = pg.TextItem(src_labels[i], color='#ff6666', anchor=(0.5, 1.5))
            txt.setFont(QFont("Segoe UI", 7))
            txt.setPos(pos[0], pos[1])
            self._plot.addItem(txt)
            self._label_items.append(txt)

        for i, pos in enumerate(det_pos):
            txt = pg.TextItem(det_labels[i], color='#6699ff', anchor=(0.5, 1.5))
            txt.setFont(QFont("Segoe UI", 7))
            txt.setPos(pos[0], pos[1])
            self._plot.addItem(txt)
            self._label_items.append(txt)


        for item in getattr(self, '_line_items', []):
            self._plot.removeItem(item)
        self._line_items = []

        for lbl in glm_hbo.pair_labels:
            parts = lbl.replace('S', '').replace('D', '').split('-')
            si, di = int(parts[0]) - 1, int(parts[1]) - 1
            line = pg.PlotDataItem(
                [src_pos[si, 0], det_pos[di, 0]],
                [src_pos[si, 1], det_pos[di, 1]],
                pen=pg.mkPen('#3e3e42', width=1),
            )
            line.setZValue(-20)
            self._plot.addItem(line)
            self._line_items.append(line)


        self._combo_cond.blockSignals(True)
        self._combo_cond.clear()
        for name in glm_hbo.contrast_names:
            self._combo_cond.addItem(name)
        self._combo_cond.blockSignals(False)

        self.setVisible(True)
        self._update_map()

    def _update_map(self):
        if self._glm_hbo is None or self._pair_midpoints is None:
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

        # Color by t-statistic; alpha by significance.
        t_abs_max = max(np.abs(t_vals).max(), 1e-6)
        t_norm = t_vals / t_abs_max  # normalized to [-1, 1]

        spots = []
        for i, (pos, t_n, p) in enumerate(zip(self._pair_midpoints, t_norm, p_vals)):
            significant = p < p_thresh

            if chrom == "HbO":
                if significant:
                    intensity = int(min(abs(t_n) * 255, 255))
                    if t_n >= 0:
                        color = pg.mkColor(255, int(255 - intensity * 0.6), 0, 200)
                    else:
                        color = pg.mkColor(0, int(255 - intensity * 0.6), 255, 200)
                else:
                    color = pg.mkColor(80, 80, 80, 60)
            else:
                if significant:
                    intensity = int(min(abs(t_n) * 255, 255))
                    if t_n <= 0:
                        color = pg.mkColor(0, int(255 - intensity * 0.4), 255, 200)
                    else:
                        color = pg.mkColor(255, int(255 - intensity * 0.6), 0, 200)
                else:
                    color = pg.mkColor(80, 80, 80, 60)

            size = 18 if significant else 10
            spots.append({
                'pos': pos, 'size': size, 'symbol': 'o',
                'pen': pg.mkPen(color, width=1),
                'brush': pg.mkBrush(color),
            })

        self._scatter.setData(spots)


        n_sig = int(np.sum(p_vals < p_thresh))
        cond_name = self._combo_cond.currentText()
        self._plot.setTitle(
            f"{chrom} — Condition '{cond_name}' — {n_sig}/{len(t_vals)} significant (p<{p_thresh:.2f})",
            color='#dcdcdc', size='10pt',
        )

    def clear(self):
        self._glm_hbo = None
        self._glm_hbr = None
        self._pair_midpoints = None
        self._scatter.clear()
        self._src_scatter.clear()
        self._det_scatter.clear()
        for item in getattr(self, '_label_items', []):
            self._plot.removeItem(item)
        for item in getattr(self, '_line_items', []):
            self._plot.removeItem(item)
        self._label_items = []
        self._line_items = []
        self.setVisible(False)
