"""
ProcessingPanel — GUI controls for the fNIRS signal processing pipeline.

Provides Auto/Manual mode toggle:
- Auto: OD + bandpass applied automatically on file load with standard defaults
- Manual: User configures parameters and clicks Apply

Always provides view switching (Raw / OD / Filtered) and Reset.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QLabel,
    QFrame, QButtonGroup,
)
from PyQt5.QtCore import pyqtSignal, Qt


# Standard fNIRS processing defaults
_DEFAULT_LOW = 0.01    # Hz — removes slow drift
_DEFAULT_HIGH = 0.1    # Hz — removes cardiac/respiratory
_DEFAULT_ORDER = 3     # Butterworth order


class ProcessingPanel(QWidget):
    """Control panel for the processing pipeline."""

    # Signals emitted to main window
    convert_od_clicked = pyqtSignal()
    apply_filter_clicked = pyqtSignal(float, float, int)  # low, high, order
    reset_clicked = pyqtSignal()
    view_raw_clicked = pyqtSignal()
    view_od_clicked = pyqtSignal()
    view_filtered_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.set_enabled(False)

    @property
    def is_auto(self) -> bool:
        """True if auto-processing mode is active."""
        return self._btn_auto.isChecked()

    @property
    def filter_low(self) -> float:
        return self._spin_low.value()

    @property
    def filter_high(self) -> float:
        return self._spin_high.value()

    @property
    def filter_order(self) -> int:
        return self._spin_order.value()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Mode Toggle: Auto / Manual ──
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setSpacing(4)

        self._btn_auto = QPushButton("Auto")
        self._btn_manual = QPushButton("Manual")
        for btn in (self._btn_auto, self._btn_manual):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { padding: 6px 12px; border-radius: 4px; "
                "background-color: #3e3e42; color: #abb2bf; font-weight: bold; }"
                "QPushButton:checked { background-color: #094771; color: #fff; }"
            )
        self._btn_auto.setChecked(True)
        self._btn_auto.setToolTip(
            "Auto: OD + bandpass applied automatically on file load\n"
            "Uses standard defaults (0.01–0.1 Hz, order 3)"
        )
        self._btn_manual.setToolTip(
            "Manual: Configure parameters and click Apply"
        )

        # Exclusive toggle
        self._btn_auto.clicked.connect(self._on_auto_clicked)
        self._btn_manual.clicked.connect(self._on_manual_clicked)

        mode_layout.addWidget(self._btn_auto)
        mode_layout.addWidget(self._btn_manual)
        layout.addWidget(mode_group)

        # ── Pipeline State Display ──
        state_group = QGroupBox("Pipeline State")
        state_layout = QVBoxLayout(state_group)
        state_layout.setSpacing(4)

        self._state_label = QLabel("No data loaded")
        self._state_label.setAlignment(Qt.AlignCenter)
        self._state_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 6px; "
            "border-radius: 4px; background-color: #3e3e42; color: #abb2bf;"
        )
        state_layout.addWidget(self._state_label)

        # State badges row
        badge_row = QHBoxLayout()
        self._badge_raw = self._make_badge("RAW", active=False)
        self._badge_od = self._make_badge("OD", active=False)
        self._badge_filt = self._make_badge("FILT", active=False)
        badge_row.addWidget(self._badge_raw)
        badge_row.addWidget(QLabel("→"))
        badge_row.addWidget(self._badge_od)
        badge_row.addWidget(QLabel("→"))
        badge_row.addWidget(self._badge_filt)
        state_layout.addLayout(badge_row)

        layout.addWidget(state_group)

        # ── Manual Controls Container ──
        self._manual_frame = QWidget()
        manual_layout = QVBoxLayout(self._manual_frame)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(6)

        # OD Conversion
        od_group = QGroupBox("1. Optical Density")
        od_layout = QVBoxLayout(od_group)

        self._btn_convert_od = QPushButton("Convert to OD")
        self._btn_convert_od.setToolTip("OD = -log₁₀(I / I₀)\nI₀ = mean of full recording")
        self._btn_convert_od.setStyleSheet(
            "QPushButton { background-color: #2d5a1a; color: #e5e5e5; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3a7a23; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_convert_od.clicked.connect(self.convert_od_clicked.emit)
        od_layout.addWidget(self._btn_convert_od)
        manual_layout.addWidget(od_group)

        # Bandpass Filter
        filter_group = QGroupBox("2. Bandpass Filter")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(4)

        # Low cutoff
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Low (Hz):"))
        self._spin_low = QDoubleSpinBox()
        self._spin_low.setRange(0.001, 1.0)
        self._spin_low.setValue(_DEFAULT_LOW)
        self._spin_low.setSingleStep(0.005)
        self._spin_low.setDecimals(3)
        self._spin_low.setToolTip("Removes slow drift below this frequency")
        row1.addWidget(self._spin_low)
        filter_layout.addLayout(row1)

        # High cutoff
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("High (Hz):"))
        self._spin_high = QDoubleSpinBox()
        self._spin_high.setRange(0.01, 5.0)
        self._spin_high.setValue(_DEFAULT_HIGH)
        self._spin_high.setSingleStep(0.01)
        self._spin_high.setDecimals(3)
        self._spin_high.setToolTip("Removes cardiac/respiratory above this frequency")
        row2.addWidget(self._spin_high)
        filter_layout.addLayout(row2)

        # Filter order
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Order:"))
        self._spin_order = QSpinBox()
        self._spin_order.setRange(1, 5)
        self._spin_order.setValue(_DEFAULT_ORDER)
        self._spin_order.setToolTip("Butterworth filter order (higher = sharper cutoff)")
        row3.addWidget(self._spin_order)
        filter_layout.addLayout(row3)

        # Apply button
        self._btn_apply_filter = QPushButton("Apply Bandpass Filter")
        self._btn_apply_filter.setStyleSheet(
            "QPushButton { background-color: #094771; color: #e5e5e5; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0d5a8c; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_apply_filter.clicked.connect(self._on_apply_filter)
        filter_layout.addWidget(self._btn_apply_filter)
        manual_layout.addWidget(filter_group)

        layout.addWidget(self._manual_frame)

        # Hide manual controls by default (Auto mode)
        self._manual_frame.setVisible(False)

        # ── View Switcher ──
        view_group = QGroupBox("View")
        view_layout = QHBoxLayout(view_group)
        view_layout.setSpacing(4)

        self._btn_view_raw = QPushButton("Raw")
        self._btn_view_od = QPushButton("OD")
        self._btn_view_filt = QPushButton("Filtered")
        for btn in (self._btn_view_raw, self._btn_view_od, self._btn_view_filt):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 8px; border-radius: 3px; "
                "background-color: #3e3e42; color: #abb2bf; }"
                "QPushButton:checked { background-color: #094771; color: #fff; }"
                "QPushButton:disabled { background-color: #2d2d30; color: #555; }"
            )
            view_layout.addWidget(btn)

        self._btn_view_raw.setChecked(True)
        self._btn_view_raw.clicked.connect(self.view_raw_clicked.emit)
        self._btn_view_od.clicked.connect(self.view_od_clicked.emit)
        self._btn_view_filt.clicked.connect(self.view_filtered_clicked.emit)

        layout.addWidget(view_group)

        # ── Reset ──
        self._btn_reset = QPushButton("Reset to Raw")
        self._btn_reset.setStyleSheet(
            "QPushButton { background-color: #5a1a1a; color: #e5e5e5; "
            "padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #7a2323; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_reset.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self._btn_reset)

        layout.addStretch()

    # ── Mode Toggle Handlers ──

    def _on_auto_clicked(self):
        self._btn_auto.setChecked(True)
        self._btn_manual.setChecked(False)
        self._manual_frame.setVisible(False)

    def _on_manual_clicked(self):
        self._btn_manual.setChecked(True)
        self._btn_auto.setChecked(False)
        self._manual_frame.setVisible(True)

    # ── Helpers ──

    def _make_badge(self, text: str, active: bool = False) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(22)
        self._style_badge(lbl, active)
        return lbl

    def _style_badge(self, lbl: QLabel, active: bool):
        if active:
            lbl.setStyleSheet(
                "background-color: #094771; color: #fff; font-size: 11px; "
                "font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
        else:
            lbl.setStyleSheet(
                "background-color: #3e3e42; color: #666; font-size: 11px; "
                "border-radius: 3px; padding: 2px 6px;"
            )

    def _on_apply_filter(self):
        low = self._spin_low.value()
        high = self._spin_high.value()
        order = self._spin_order.value()
        self.apply_filter_clicked.emit(low, high, order)

    # ── Public API ──

    def set_enabled(self, enabled: bool):
        """Enable/disable all controls."""
        self._btn_convert_od.setEnabled(enabled)
        self._btn_apply_filter.setEnabled(enabled)
        self._btn_reset.setEnabled(enabled)
        self._btn_view_raw.setEnabled(enabled)
        self._btn_view_od.setEnabled(False)
        self._btn_view_filt.setEnabled(False)
        if not enabled:
            self._state_label.setText("No data loaded")
            self._update_badges("RAW")

    def update_state(self, state_label: str, has_od: bool, has_filtered: bool):
        """Update the panel to reflect the current pipeline state."""
        self._state_label.setText(state_label)

        # Enable/disable view buttons
        self._btn_view_raw.setEnabled(True)
        self._btn_view_od.setEnabled(has_od)
        self._btn_view_filt.setEnabled(has_filtered)

        # Update checked states
        self._btn_view_raw.setChecked(state_label == "Raw Intensity")
        self._btn_view_od.setChecked(state_label == "Optical Density")
        self._btn_view_filt.setChecked(state_label == "Filtered OD")

        # Update badges
        if "Filtered" in state_label:
            self._update_badges("FILT")
        elif "Optical" in state_label:
            self._update_badges("OD")
        else:
            self._update_badges("RAW")

    def _update_badges(self, active: str):
        self._style_badge(self._badge_raw, active in ("RAW", "OD", "FILT"))
        self._style_badge(self._badge_od, active in ("OD", "FILT"))
        self._style_badge(self._badge_filt, active == "FILT")
