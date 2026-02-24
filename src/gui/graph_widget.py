"""
GraphWidget — High-performance multi-channel fNIRS time-series viewer.

Features:
- Channels grouped by source-detector pair (each pair shows both wavelengths)
- Wavelength filter toggle (WL1 / WL2 / Both)
- Channel quality flags (CV-based, highlights noisy or flat channels)
- Stacked (offset) view vs overlaid view
- Downsampling + clip-to-view for smooth zoom/pan
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QScrollArea, QPushButton, QSpinBox,
    QFrame, QButtonGroup, QRadioButton, QToolButton,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from data_io.snirf_loader_base import SNIRFData


# ── Performance config ───────────────────────
pg.setConfigOptions(
    antialias=False,
    background='#1e1e1e',
    foreground='#dcdcdc',
    useOpenGL=False,
)

# Wavelength colours (convention: red-ish for HbO wavelength, blue-ish for HbR)
_WL_COLORS = {
    1: "#e06c75",   # wavelength 1 (typically 760nm → deoxy)
    2: "#61afef",   # wavelength 2 (typically 850nm → oxy)
}
_WL_COLORS_DIM = {
    1: "#7a3a3f",
    2: "#2e5a7a",
}

# Quality thresholds
_CV_BAD_THRESHOLD = 0.50    # coefficient of variation > 50% → bad
_CV_FLAT_THRESHOLD = 0.001  # CV < 0.1% → flat / no contact

_DEFAULT_PAIRS_SHOWN = 5


class _PairWidget(QFrame):
    """A collapsible row for one S-D pair with wavelength sub-checkboxes."""

    toggled = pyqtSignal()  # emitted when any checkbox changes

    def __init__(self, pair_key: str, wl_items: list[dict], parent=None):
        super().__init__(parent)
        self.pair_key = pair_key
        self._wl_items = wl_items   # [{"wl_idx", "ch_idx", "wl_nm", "quality"}]

        self.setStyleSheet("""
            _PairWidget {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                background: #252526;
                margin: 1px 2px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)

        # ── Header row: pair checkbox + quality badge ──
        header = QHBoxLayout()
        self.pair_cb = QCheckBox(pair_key)
        self.pair_cb.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.pair_cb.setStyleSheet("color: #dcdcdc;")
        self.pair_cb.toggled.connect(self._on_pair_toggled)
        header.addWidget(self.pair_cb)

        # Quality badge
        worst_q = max(item["quality"] for item in wl_items)
        if worst_q == 2:
            badge = QLabel("⚠ BAD")
            badge.setStyleSheet("color: #e06c75; font-size: 9px; font-weight: bold;")
        elif worst_q == 1:
            badge = QLabel("▬ FLAT")
            badge.setStyleSheet("color: #e5c07b; font-size: 9px; font-weight: bold;")
        else:
            badge = QLabel("● OK")
            badge.setStyleSheet("color: #98c379; font-size: 9px; font-weight: bold;")
        header.addStretch()
        header.addWidget(badge)
        layout.addLayout(header)

        # ── Wavelength sub-checkboxes ──
        self.wl_cbs: dict[int, QCheckBox] = {}
        for item in wl_items:
            wl_idx = item["wl_idx"]
            color = _WL_COLORS.get(wl_idx, "#abb2bf")
            cb = QCheckBox(f"  {item['wl_nm']:.0f} nm")
            cb.setStyleSheet(f"color: {color}; font-size: 10px; margin-left: 16px;")
            cb.toggled.connect(self._on_wl_toggled)
            self.wl_cbs[wl_idx] = cb
            layout.addWidget(cb)

    def set_checked(self, checked: bool):
        """Check/uncheck the entire pair."""
        self.pair_cb.blockSignals(True)
        self.pair_cb.setChecked(checked)
        self.pair_cb.blockSignals(False)
        for cb in self.wl_cbs.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.toggled.emit()

    def set_wavelength_filter(self, wl_idx: int | None):
        """Show only specific wavelength (None=both)."""
        for idx, cb in self.wl_cbs.items():
            cb.blockSignals(True)
            if wl_idx is None:
                cb.setChecked(self.pair_cb.isChecked())
            else:
                cb.setChecked(idx == wl_idx and self.pair_cb.isChecked())
            cb.blockSignals(False)
        self.toggled.emit()

    def get_visible_channels(self) -> list[int]:
        """Return list of channel indices that should be visible."""
        result = []
        for item in self._wl_items:
            wl_idx = item["wl_idx"]
            if wl_idx in self.wl_cbs and self.wl_cbs[wl_idx].isChecked():
                result.append(item["ch_idx"])
        return result

    def _on_pair_toggled(self, checked):
        for cb in self.wl_cbs.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.toggled.emit()

    def _on_wl_toggled(self):
        any_on = any(cb.isChecked() for cb in self.wl_cbs.values())
        self.pair_cb.blockSignals(True)
        self.pair_cb.setChecked(any_on)
        self.pair_cb.blockSignals(False)
        self.toggled.emit()


class GraphWidget(QWidget):
    """Interactive fNIRS time-series viewer with channel grouping and quality flags."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curves: dict[int, pg.PlotDataItem] = {}    # ch_idx → curve
        self._pair_widgets: list[_PairWidget] = []
        self._data: SNIRFData | None = None
        self._stacked = False
        self._wl_filter: int | None = None    # None = both
        self._quality: dict[int, int] = {}    # ch_idx → 0=ok, 1=flat, 2=bad
        self._build_ui()

    # ══════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── Plot area ──
        plot_container = QVBoxLayout()

        # Title bar
        title_bar = QHBoxLayout()
        self._title = QLabel("  Raw Optical Intensity")
        self._title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._title.setStyleSheet("color: #dcdcdc;")
        title_bar.addWidget(self._title)

        # Stacked/Overlaid toggle
        self._btn_stacked = QPushButton("📊 Stacked")
        self._btn_overlaid = QPushButton("📈 Overlaid")
        for btn in (self._btn_stacked, self._btn_overlaid):
            btn.setFixedHeight(26)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3e3e42; color: #dcdcdc; border: none;
                    border-radius: 3px; padding: 2px 10px; font-size: 11px;
                }
                QPushButton:checked { background-color: #007acc; color: white; }
                QPushButton:hover { background-color: #505054; }
            """)
        self._btn_overlaid.setChecked(True)
        self._btn_stacked.clicked.connect(lambda: self._set_view_mode(True))
        self._btn_overlaid.clicked.connect(lambda: self._set_view_mode(False))
        title_bar.addStretch()
        title_bar.addWidget(self._btn_overlaid)
        title_bar.addWidget(self._btn_stacked)

        title_frame = QFrame()
        title_frame.setLayout(title_bar)
        title_frame.setFixedHeight(36)
        title_frame.setStyleSheet("background-color: #2d2d30; border-bottom: 1px solid #3e3e42;")
        plot_container.addWidget(title_frame)

        # PyQtGraph PlotWidget
        self._plot = pg.PlotWidget()
        self._plot.setLabel('bottom', 'Time', units='s')
        self._plot.setLabel('left', 'Intensity', units='a.u.')
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.getPlotItem().getViewBox().setAutoVisible(y=True)
        plot_container.addWidget(self._plot)

        main_layout.addLayout(plot_container, stretch=5)

        # ── Sidebar ──
        sidebar = QVBoxLayout()
        sidebar.setSpacing(4)

        # Header
        sidebar_header = QLabel("  Channels")
        sidebar_header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sidebar_header.setFixedHeight(36)
        sidebar_header.setStyleSheet(
            "background-color: #2d2d30; color: #dcdcdc; border-bottom: 1px solid #3e3e42;"
        )
        sidebar.addWidget(sidebar_header)

        # ── Wavelength filter ──
        wl_frame = QFrame()
        wl_frame.setStyleSheet("background: #252526; border-radius: 4px; padding: 2px;")
        wl_layout = QHBoxLayout(wl_frame)
        wl_layout.setContentsMargins(6, 4, 6, 4)
        wl_label = QLabel("λ Filter:")
        wl_label.setStyleSheet("color: #888; font-size: 10px;")
        wl_layout.addWidget(wl_label)

        self._btn_wl_both = QPushButton("Both")
        self._btn_wl1 = QPushButton("λ₁")
        self._btn_wl2 = QPushButton("λ₂")
        for btn in (self._btn_wl_both, self._btn_wl1, self._btn_wl2):
            btn.setFixedHeight(22)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background: #3e3e42; color: #dcdcdc; border: none;
                    border-radius: 3px; padding: 1px 8px; font-size: 10px;
                }
                QPushButton:checked { background: #007acc; color: white; }
                QPushButton:hover { background: #505054; }
            """)
        self._btn_wl_both.setChecked(True)
        self._btn_wl_both.clicked.connect(lambda: self._set_wl_filter(None))
        self._btn_wl1.clicked.connect(lambda: self._set_wl_filter(1))
        self._btn_wl2.clicked.connect(lambda: self._set_wl_filter(2))
        wl_layout.addWidget(self._btn_wl_both)
        wl_layout.addWidget(self._btn_wl1)
        wl_layout.addWidget(self._btn_wl2)
        sidebar.addWidget(wl_frame)

        # ── Quality summary label ──
        self._quality_label = QLabel("")
        self._quality_label.setStyleSheet("color: #888; font-size: 10px; padding: 0 6px;")
        self._quality_label.setWordWrap(True)
        sidebar.addWidget(self._quality_label)

        # ── Pair selection buttons ──
        btn_row = QHBoxLayout()
        self._btn_all = QPushButton("All")
        self._btn_none = QPushButton("None")
        self._btn_first = QPushButton("First:")
        self._spin_first = QSpinBox()
        self._spin_first.setRange(1, 200)
        self._spin_first.setValue(_DEFAULT_PAIRS_SHOWN)
        self._spin_first.setFixedWidth(50)

        for btn in (self._btn_all, self._btn_none, self._btn_first):
            btn.setFixedHeight(24)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3e3e42; color: #dcdcdc; border: none;
                    border-radius: 3px; padding: 2px 6px; font-size: 11px;
                }
                QPushButton:hover { background-color: #505054; }
            """)
        self._spin_first.setStyleSheet("""
            QSpinBox {
                background: #2d2d30; color: #dcdcdc; border: 1px solid #3e3e42;
                border-radius: 3px; padding: 1px 4px; font-size: 11px;
            }
        """)

        self._btn_all.clicked.connect(self._select_all)
        self._btn_none.clicked.connect(self._select_none)
        self._btn_first.clicked.connect(self._select_first_n)

        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        btn_row.addWidget(self._btn_first)
        btn_row.addWidget(self._spin_first)
        sidebar.addLayout(btn_row)

        # ── Scrollable pair list ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(210)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._pair_container = QWidget()
        self._pair_layout = QVBoxLayout(self._pair_container)
        self._pair_layout.setAlignment(Qt.AlignTop)
        self._pair_layout.setSpacing(2)
        self._pair_layout.setContentsMargins(2, 2, 2, 2)
        scroll.setWidget(self._pair_container)
        sidebar.addWidget(scroll)

        main_layout.addLayout(sidebar, stretch=1)

    # ══════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════

    def plot_data(self, data: SNIRFData):
        """Load channel data, compute quality, build pair list, and plot."""
        self._data = data
        self._plot.clear()
        self._curves.clear()
        self._clear_pair_list()
        self._wl_filter = None
        self._stacked = False
        self._btn_overlaid.setChecked(True)
        self._btn_stacked.setChecked(False)
        self._btn_wl_both.setChecked(True)
        self._btn_wl1.setChecked(False)
        self._btn_wl2.setChecked(False)

        # Update wavelength button labels with actual values
        wls = data.probe.wavelengths
        if len(wls) >= 1:
            self._btn_wl1.setText(f"{wls[0]:.0f}")
        if len(wls) >= 2:
            self._btn_wl2.setText(f"{wls[1]:.0f}")

        # ── Compute channel quality ──
        self._compute_quality(data)

        # ── Group channels by S-D pair ──
        pair_map: OrderedDict[str, list[dict]] = OrderedDict()
        for ch_idx, ch in enumerate(data.channels):
            pair_key = f"S{ch.source_index}–D{ch.detector_index}"
            wl_nm = data.probe.wavelengths[ch.wavelength_index - 1]
            item = {
                "wl_idx": ch.wavelength_index,
                "ch_idx": ch_idx,
                "wl_nm": wl_nm,
                "quality": self._quality.get(ch_idx, 0),
            }
            pair_map.setdefault(pair_key, []).append(item)

        # ── Create curves for all channels ──
        downsample = max(1, len(data.time) // 2000)
        for ch_idx in range(data.n_channels):
            ch = data.channels[ch_idx]
            wl_idx = ch.wavelength_index
            color = _WL_COLORS.get(wl_idx, "#abb2bf")

            curve = pg.PlotDataItem(
                pen=pg.mkPen(color, width=1),
                skipFiniteCheck=True,
                clipToView=True,
                downsample=downsample,
                downsampleMethod='peak',
            )
            curve.setData([], [])  # start hidden
            self._plot.addItem(curve)
            self._curves[ch_idx] = curve

        # ── Build pair widgets ──
        n_good = 0
        n_flat = 0
        n_bad = 0
        for pair_idx, (pair_key, wl_items) in enumerate(pair_map.items()):
            pw = _PairWidget(pair_key, wl_items)
            pw.toggled.connect(self._refresh_curves)
            self._pair_layout.addWidget(pw)
            self._pair_widgets.append(pw)

            # Auto-enable first N pairs
            if pair_idx < _DEFAULT_PAIRS_SHOWN:
                pw.set_checked(True)

            # Count quality
            worst = max(item["quality"] for item in wl_items)
            if worst == 2:
                n_bad += 1
            elif worst == 1:
                n_flat += 1
            else:
                n_good += 1

        # Quality summary
        total = len(pair_map)
        self._quality_label.setText(
            f"Quality: {n_good}/{total} OK  ·  {n_flat} flat  ·  {n_bad} noisy"
        )

        self._title.setText(
            f"  Raw Intensity — {data.n_channels} ch, {total} pairs"
        )

        self._plot_array = data.intensity  # active data for plotting
        self._refresh_curves()

    def update_data(self, new_array, snirf_data, title_prefix: str):
        """Re-plot using processed data (OD / filtered) without rebuilding pair list.

        Parameters
        ----------
        new_array : ndarray, shape (n_time, n_ch)
            Processed data to display.
        snirf_data : SNIRFData
            Original data (for time axis and channel info).
        title_prefix : str
            Label like 'Optical Density' or 'Filtered OD'.
        """
        self._data = snirf_data
        self._plot_array = new_array
        total = len(self._pair_widgets)
        self._title.setText(
            f"  {title_prefix} — {snirf_data.n_channels} ch, {total} pairs"
        )
        self._refresh_curves()

    def clear_plot(self):
        self._plot.clear()
        self._curves.clear()
        self._clear_pair_list()
        self._title.setText("  Raw Optical Intensity")
        self._quality_label.setText("")
        self._data = None

    # ══════════════════════════════════════════
    #  Quality Computation
    # ══════════════════════════════════════════

    def _compute_quality(self, data: SNIRFData):
        """Compute per-channel quality using coefficient of variation."""
        self._quality.clear()
        for ch_idx in range(data.n_channels):
            signal = data.intensity[:, ch_idx]
            mean = np.mean(signal)
            if mean == 0:
                self._quality[ch_idx] = 2  # bad
                continue
            cv = np.std(signal) / abs(mean)
            if cv < _CV_FLAT_THRESHOLD:
                self._quality[ch_idx] = 1  # flat
            elif cv > _CV_BAD_THRESHOLD:
                self._quality[ch_idx] = 2  # bad
            else:
                self._quality[ch_idx] = 0  # ok

    # ══════════════════════════════════════════
    #  Curve Refresh (called on any toggle)
    # ══════════════════════════════════════════

    def _refresh_curves(self):
        """Update which curves have data based on pair checkboxes."""
        if self._data is None:
            return

        plot_arr = getattr(self, '_plot_array', self._data.intensity)

        # Collect all visible channel indices
        visible_set: set[int] = set()
        for pw in self._pair_widgets:
            visible_set.update(pw.get_visible_channels())

        # Calculate offset for stacked view
        visible_list = sorted(visible_set)
        offsets: dict[int, float] = {}
        if self._stacked and visible_list:
            sample = plot_arr[:, visible_list[0]]
            offset_step = (np.max(sample) - np.min(sample)) * 1.2
            if offset_step == 0:
                offset_step = 1.0
            for rank, ch_idx in enumerate(visible_list):
                offsets[ch_idx] = rank * offset_step

        # Apply data to curves
        time = self._data.time
        for ch_idx, curve in self._curves.items():
            if ch_idx in visible_set:
                y = plot_arr[:, ch_idx].copy()
                if ch_idx in offsets:
                    y = y - np.mean(y) + offsets[ch_idx]
                curve.setData(time, y)
            else:
                curve.setData([], [])

        if self._stacked and visible_list:
            self._plot.setLabel('left', 'Channel (stacked)', units='')
        else:
            # Adjust label based on data type
            if hasattr(self, '_plot_array') and self._plot_array is not self._data.intensity:
                self._plot.setLabel('left', 'OD / Filtered', units='')
            else:
                self._plot.setLabel('left', 'Intensity', units='a.u.')

        try:
            self._plot.getPlotItem().getViewBox().autoRange()
        except Exception:
            pass

    # ══════════════════════════════════════════
    #  Controls
    # ══════════════════════════════════════════

    def _set_view_mode(self, stacked: bool):
        self._stacked = stacked
        self._btn_stacked.setChecked(stacked)
        self._btn_overlaid.setChecked(not stacked)
        self._refresh_curves()

    def _set_wl_filter(self, wl_idx: int | None):
        self._wl_filter = wl_idx
        self._btn_wl_both.setChecked(wl_idx is None)
        self._btn_wl1.setChecked(wl_idx == 1)
        self._btn_wl2.setChecked(wl_idx == 2)
        for pw in self._pair_widgets:
            pw.set_wavelength_filter(wl_idx)

    def _select_all(self):
        for pw in self._pair_widgets:
            pw.set_checked(True)
        if self._wl_filter is not None:
            self._set_wl_filter(self._wl_filter)
        self._refresh_curves()

    def _select_none(self):
        for pw in self._pair_widgets:
            pw.set_checked(False)
        self._refresh_curves()

    def _select_first_n(self):
        n = self._spin_first.value()
        for i, pw in enumerate(self._pair_widgets):
            pw.set_checked(i < n)
        if self._wl_filter is not None:
            self._set_wl_filter(self._wl_filter)
        self._refresh_curves()

    def _clear_pair_list(self):
        for pw in self._pair_widgets:
            pw.deleteLater()
        self._pair_widgets.clear()
