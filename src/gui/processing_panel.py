"""ProcessingPanel — GUI controls for the fNIRS processing pipeline."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QLabel,
    QFrame, QButtonGroup, QComboBox,
)
from PyQt5.QtCore import pyqtSignal, Qt

_DEFAULT_LOW = 0.01
_DEFAULT_HIGH = 0.1
_DEFAULT_ORDER = 3


class ProcessingPanel(QWidget):
    """Control panel for the processing pipeline."""

    convert_od_clicked = pyqtSignal()
    apply_correction_clicked = pyqtSignal(str)
    apply_filter_clicked = pyqtSignal(float, float, int)
    convert_conc_clicked = pyqtSignal()
    apply_all_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    view_raw_clicked = pyqtSignal()
    view_od_clicked = pyqtSignal()
    view_corrected_clicked = pyqtSignal()
    view_filtered_clicked = pyqtSignal()
    view_conc_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.set_enabled(False)

    @property
    def is_auto(self) -> bool:
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

    @property
    def correction_method(self) -> str:
        return self._combo_method.currentText().lower()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

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
        self._btn_auto.setToolTip("Auto: OD + TDDR + bandpass + MBLL on file load")
        self._btn_manual.setToolTip("Manual: configure and apply step-by-step")
        self._btn_auto.clicked.connect(self._on_auto_clicked)
        self._btn_manual.clicked.connect(self._on_manual_clicked)
        mode_layout.addWidget(self._btn_auto)
        mode_layout.addWidget(self._btn_manual)
        layout.addWidget(mode_group)

        self._btn_apply_all = QPushButton("▶  Apply All  (OD → TDDR → Filter → HbO/HbR)")
        self._btn_apply_all.setStyleSheet(
            "QPushButton { background-color: #1a6b1a; color: #fff; "
            "padding: 10px; border-radius: 5px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background-color: #22881f; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_apply_all.setToolTip(
            "Run full pipeline: OD → TDDR correction → Bandpass → MBLL (HbO/HbR)"
        )
        self._btn_apply_all.clicked.connect(self.apply_all_clicked.emit)
        layout.addWidget(self._btn_apply_all)

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

        badge_row = QHBoxLayout()
        self._badge_raw = self._make_badge("RAW")
        self._badge_od = self._make_badge("OD")
        self._badge_corr = self._make_badge("CORR")
        self._badge_filt = self._make_badge("FILT")
        self._badge_conc = self._make_badge("HbO/R")
        for b in (self._badge_raw, self._badge_od, self._badge_corr,
                  self._badge_filt, self._badge_conc):
            badge_row.addWidget(b)
        state_layout.addLayout(badge_row)
        layout.addWidget(state_group)

        self._manual_frame = QWidget()
        manual_layout = QVBoxLayout(self._manual_frame)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(6)

        od_group = QGroupBox("1. Optical Density")
        od_layout = QVBoxLayout(od_group)
        self._btn_convert_od = QPushButton("Convert to OD")
        self._btn_convert_od.setToolTip("OD = -log₁₀(I / I₀)")
        self._btn_convert_od.setStyleSheet(
            "QPushButton { background-color: #2d5a1a; color: #e5e5e5; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3a7a23; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_convert_od.clicked.connect(self.convert_od_clicked.emit)
        od_layout.addWidget(self._btn_convert_od)
        manual_layout.addWidget(od_group)

        mc_group = QGroupBox("2. Motion Correction")
        mc_layout = QVBoxLayout(mc_group)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._combo_method = QComboBox()
        self._combo_method.addItems(["TDDR", "Spline"])
        self._combo_method.setToolTip(
            "TDDR: robust derivative repair (recommended)\n"
            "Spline: cubic spline over detected artifacts"
        )
        method_row.addWidget(self._combo_method)
        mc_layout.addLayout(method_row)
        self._btn_correct = QPushButton("Apply Correction")
        self._btn_correct.setStyleSheet(
            "QPushButton { background-color: #5a4a1a; color: #e5e5e5; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #7a6a23; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_correct.clicked.connect(self._on_apply_correction)
        mc_layout.addWidget(self._btn_correct)
        manual_layout.addWidget(mc_group)

        filter_group = QGroupBox("3. Bandpass Filter")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Low (Hz):"))
        self._spin_low = QDoubleSpinBox()
        self._spin_low.setRange(0.001, 1.0)
        self._spin_low.setValue(_DEFAULT_LOW)
        self._spin_low.setSingleStep(0.005)
        self._spin_low.setDecimals(3)
        row1.addWidget(self._spin_low)
        filter_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("High (Hz):"))
        self._spin_high = QDoubleSpinBox()
        self._spin_high.setRange(0.01, 5.0)
        self._spin_high.setValue(_DEFAULT_HIGH)
        self._spin_high.setSingleStep(0.01)
        self._spin_high.setDecimals(3)
        row2.addWidget(self._spin_high)
        filter_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Order:"))
        self._spin_order = QSpinBox()
        self._spin_order.setRange(1, 5)
        self._spin_order.setValue(_DEFAULT_ORDER)
        row3.addWidget(self._spin_order)
        filter_layout.addLayout(row3)

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

        conc_group = QGroupBox("4. MBLL (HbO / HbR)")
        conc_layout = QVBoxLayout(conc_group)
        self._btn_convert_conc = QPushButton("Convert to HbO / HbR")
        self._btn_convert_conc.setToolTip("Modified Beer-Lambert Law\nOD → ΔHbO + ΔHbR (μmol/L)")
        self._btn_convert_conc.setStyleSheet(
            "QPushButton { background-color: #5a1a5a; color: #e5e5e5; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #7a237a; }"
            "QPushButton:disabled { background-color: #2d2d30; color: #666; }"
        )
        self._btn_convert_conc.clicked.connect(self.convert_conc_clicked.emit)
        conc_layout.addWidget(self._btn_convert_conc)
        manual_layout.addWidget(conc_group)

        layout.addWidget(self._manual_frame)
        self._manual_frame.setVisible(False)

        view_group = QGroupBox("View")
        view_layout = QHBoxLayout(view_group)
        view_layout.setSpacing(3)

        self._btn_view_raw = QPushButton("Raw")
        self._btn_view_od = QPushButton("OD")
        self._btn_view_corr = QPushButton("Corr")
        self._btn_view_filt = QPushButton("Filt")
        self._btn_view_conc = QPushButton("HbO/R")
        for btn in (self._btn_view_raw, self._btn_view_od, self._btn_view_corr,
                    self._btn_view_filt, self._btn_view_conc):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 5px; border-radius: 3px; "
                "background-color: #3e3e42; color: #abb2bf; font-size: 11px; }"
                "QPushButton:checked { background-color: #094771; color: #fff; }"
                "QPushButton:disabled { background-color: #2d2d30; color: #555; }"
            )
            view_layout.addWidget(btn)

        self._btn_view_raw.setChecked(True)
        self._btn_view_raw.clicked.connect(self.view_raw_clicked.emit)
        self._btn_view_od.clicked.connect(self.view_od_clicked.emit)
        self._btn_view_corr.clicked.connect(self.view_corrected_clicked.emit)
        self._btn_view_filt.clicked.connect(self.view_filtered_clicked.emit)
        self._btn_view_conc.clicked.connect(self.view_conc_clicked.emit)
        layout.addWidget(view_group)

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

    def _on_auto_clicked(self):
        self._btn_auto.setChecked(True)
        self._btn_manual.setChecked(False)
        self._manual_frame.setVisible(False)
        self._btn_apply_all.setVisible(True)

    def _on_manual_clicked(self):
        self._btn_manual.setChecked(True)
        self._btn_auto.setChecked(False)
        self._manual_frame.setVisible(True)
        self._btn_apply_all.setVisible(False)

    def _make_badge(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(22)
        self._style_badge(lbl, False)
        return lbl

    def _style_badge(self, lbl: QLabel, active: bool):
        if active:
            lbl.setStyleSheet(
                "background-color: #094771; color: #fff; font-size: 10px; "
                "font-weight: bold; border-radius: 3px; padding: 2px 4px;"
            )
        else:
            lbl.setStyleSheet(
                "background-color: #3e3e42; color: #666; font-size: 10px; "
                "border-radius: 3px; padding: 2px 4px;"
            )

    def _on_apply_filter(self):
        self.apply_filter_clicked.emit(
            self._spin_low.value(), self._spin_high.value(), self._spin_order.value(),
        )

    def _on_apply_correction(self):
        self.apply_correction_clicked.emit(self._combo_method.currentText().lower())

    def set_enabled(self, enabled: bool):
        """Enable/disable all controls."""
        self._btn_convert_od.setEnabled(enabled)
        self._btn_correct.setEnabled(enabled)
        self._btn_apply_filter.setEnabled(enabled)
        self._btn_convert_conc.setEnabled(enabled)
        self._btn_apply_all.setEnabled(enabled)
        self._btn_reset.setEnabled(enabled)
        self._btn_view_raw.setEnabled(enabled)
        self._btn_view_od.setEnabled(False)
        self._btn_view_corr.setEnabled(False)
        self._btn_view_filt.setEnabled(False)
        self._btn_view_conc.setEnabled(False)
        if not enabled:
            self._state_label.setText("No data loaded")
            self._update_badges("RAW")

    def update_state(
        self, state_label: str,
        has_od: bool, has_filtered: bool,
        has_corrected: bool = False,
        has_concentration: bool = False,
    ):
        """Update panel to reflect current pipeline state."""
        self._state_label.setText(state_label)

        self._btn_view_raw.setEnabled(True)
        self._btn_view_od.setEnabled(has_od)
        self._btn_view_corr.setEnabled(has_corrected)
        self._btn_view_filt.setEnabled(has_filtered)
        self._btn_view_conc.setEnabled(has_concentration)

        self._btn_view_raw.setChecked(state_label == "Raw Intensity")
        self._btn_view_od.setChecked(state_label == "Optical Density")
        self._btn_view_corr.setChecked(state_label == "Motion Corrected")
        self._btn_view_filt.setChecked(state_label == "Filtered OD")
        self._btn_view_conc.setChecked(state_label == "HbO / HbR")

        if "HbO" in state_label:
            self._update_badges("CONC")
        elif "Filtered" in state_label:
            self._update_badges("FILT")
        elif "Corrected" in state_label:
            self._update_badges("CORR")
        elif "Optical" in state_label:
            self._update_badges("OD")
        else:
            self._update_badges("RAW")

    def _update_badges(self, active: str):
        self._style_badge(self._badge_raw, active in ("RAW", "OD", "CORR", "FILT", "CONC"))
        self._style_badge(self._badge_od, active in ("OD", "CORR", "FILT", "CONC"))
        self._style_badge(self._badge_corr, active in ("CORR", "FILT", "CONC"))
        self._style_badge(self._badge_filt, active in ("FILT", "CONC"))
        self._style_badge(self._badge_conc, active == "CONC")
