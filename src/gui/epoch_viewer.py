"""EpochViewer — Block-averaged HRF visualization panel."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QDoubleSpinBox, QGroupBox,
    QFrame,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

import numpy as np
import pyqtgraph as pg

from processing.epoch_extraction import BlockAverageResult

_HBO_COLOR = (224, 108, 117)
_HBR_COLOR = (97, 175, 239)
_HBO_SEM_COLOR = (224, 108, 117, 50)
_HBR_SEM_COLOR = (97, 175, 239, 50)


class EpochViewer(QWidget):
    """Panel for averaged HRF visualization with SEM shading."""

    compute_requested = pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: dict[str, BlockAverageResult] = {}
        self._current_pair_idx: int = 0
        self._build_ui()
        self.set_enabled(False)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._title = QLabel("  Block-Averaged HRF")
        self._title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._title.setFixedHeight(32)
        self._title.setStyleSheet(
            "background-color: #1e1e1e; color: #dcdcdc; "
            "border-bottom: 2px solid #007acc; padding-left: 8px;"
        )
        layout.addWidget(self._title)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QLabel("Condition:"))
        self._combo_condition = QComboBox()
        self._combo_condition.setMinimumWidth(120)
        self._combo_condition.currentTextChanged.connect(self._on_condition_changed)
        controls.addWidget(self._combo_condition)

        controls.addWidget(QLabel("Pair:"))
        self._combo_pair = QComboBox()
        self._combo_pair.setMinimumWidth(80)
        self._combo_pair.currentIndexChanged.connect(self._on_pair_changed)
        controls.addWidget(self._combo_pair)

        controls.addWidget(QLabel("Pre (s):"))
        self._spin_pre = QDoubleSpinBox()
        self._spin_pre.setRange(0.5, 10.0)
        self._spin_pre.setValue(2.0)
        self._spin_pre.setSingleStep(0.5)
        self._spin_pre.setFixedWidth(60)
        controls.addWidget(self._spin_pre)

        controls.addWidget(QLabel("Post (s):"))
        self._spin_post = QDoubleSpinBox()
        self._spin_post.setRange(5.0, 60.0)
        self._spin_post.setValue(20.0)
        self._spin_post.setSingleStep(1.0)
        self._spin_post.setFixedWidth(60)
        controls.addWidget(self._spin_post)

        self._btn_compute = QPushButton("Compute")
        self._btn_compute.setStyleSheet(
            "QPushButton { background-color: #094771; color: #fff; "
            "padding: 4px 12px; border-radius: 3px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0d5a8c; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_compute.clicked.connect(self._on_compute_clicked)
        controls.addWidget(self._btn_compute)

        controls.addStretch()

        self._trials_label = QLabel("")
        self._trials_label.setStyleSheet("color: #888; font-size: 11px;")
        controls.addWidget(self._trials_label)

        layout.addLayout(controls)

        self._plot = pg.PlotWidget()
        self._plot.setBackground('#1e1e1e')
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel('left', 'Concentration', units='μmol/L')
        self._plot.setLabel('bottom', 'Time', units='s')
        self._plot.getPlotItem().getAxis('left').setPen('#888')
        self._plot.getPlotItem().getAxis('bottom').setPen('#888')

        self._onset_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen('#ffcc00', width=1.5, style=pg.QtCore.Qt.DashLine),
            label='Onset', labelOpts={'color': '#ffcc00', 'position': 0.95},
        )
        self._plot.addItem(self._onset_line)

        layout.addWidget(self._plot, stretch=1)

        legend_row = QHBoxLayout()
        for color, label in [('#e06c75', '● HbO (mean ± SEM)'), ('#61afef', '● HbR (mean ± SEM)')]:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; padding: 2px 8px;")
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        layout.addLayout(legend_row)

    def set_enabled(self, enabled: bool):
        self._combo_condition.setEnabled(enabled)
        self._combo_pair.setEnabled(enabled)
        self._spin_pre.setEnabled(enabled)
        self._spin_post.setEnabled(enabled)
        self._btn_compute.setEnabled(enabled)

    def set_conditions(self, condition_names: list[str]):
        """Populate the condition dropdown."""
        self._combo_condition.blockSignals(True)
        self._combo_condition.clear()
        self._combo_condition.addItems(condition_names)
        self._combo_condition.blockSignals(False)
        self.set_enabled(len(condition_names) > 0)

    def set_results(self, results: list[BlockAverageResult]):
        """Store block average results and update display."""
        self._results.clear()
        for r in results:
            self._results[r.condition] = r

        if results:
            self._combo_pair.blockSignals(True)
            self._combo_pair.clear()
            self._combo_pair.addItems(results[0].pair_labels)
            self._combo_pair.blockSignals(False)

        self._update_plot()

    def _on_condition_changed(self, text: str):
        self._update_plot()

    def _on_pair_changed(self, idx: int):
        self._current_pair_idx = max(0, idx)
        self._update_plot()

    def _on_compute_clicked(self):
        condition = self._combo_condition.currentText()
        if condition:
            self.compute_requested.emit(
                condition, self._spin_pre.value(), self._spin_post.value(),
            )

    def _update_plot(self):
        self._plot.clear()
        self._plot.addItem(self._onset_line)

        condition = self._combo_condition.currentText()
        if not condition or condition not in self._results:
            return

        result = self._results[condition]
        pair_idx = min(self._current_pair_idx, result.hbo_mean.shape[1] - 1)

        t = result.epoch_time
        hbo_mean = result.hbo_mean[:, pair_idx]
        hbo_sem = result.hbo_sem[:, pair_idx]
        hbr_mean = result.hbr_mean[:, pair_idx]
        hbr_sem = result.hbr_sem[:, pair_idx]

        fill_hbo = pg.FillBetweenItem(
            pg.PlotDataItem(t, hbo_mean + hbo_sem),
            pg.PlotDataItem(t, hbo_mean - hbo_sem),
            brush=pg.mkBrush(*_HBO_SEM_COLOR),
        )
        self._plot.addItem(fill_hbo)

        fill_hbr = pg.FillBetweenItem(
            pg.PlotDataItem(t, hbr_mean + hbr_sem),
            pg.PlotDataItem(t, hbr_mean - hbr_sem),
            brush=pg.mkBrush(*_HBR_SEM_COLOR),
        )
        self._plot.addItem(fill_hbr)

        self._plot.plot(t, hbo_mean, pen=pg.mkPen(color=_HBO_COLOR, width=2), name='HbO')
        self._plot.plot(t, hbr_mean, pen=pg.mkPen(color=_HBR_COLOR, width=2), name='HbR')

        pair_label = result.pair_labels[pair_idx] if pair_idx < len(result.pair_labels) else ""
        self._title.setText(f"  Block-Averaged HRF — {condition} — {pair_label}")
        self._trials_label.setText(f"{result.n_trials} trials averaged")

    def clear(self):
        self._plot.clear()
        self._plot.addItem(self._onset_line)
        self._results.clear()
        self._combo_condition.clear()
        self._combo_pair.clear()
        self._trials_label.setText("")
        self._title.setText("  Block-Averaged HRF")
        self.set_enabled(False)
