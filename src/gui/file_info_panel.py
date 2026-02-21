"""FileInfoPanel — Displays metadata about the loaded SNIRF file."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox,
    QFormLayout, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from data_io.snirf_loader_base import SNIRFData


class FileInfoPanel(QWidget):
    """Left-side panel showing loaded file information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self._build_ui()
        self._show_placeholder()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("  File Information")
        header.setFont(QFont("Segoe UI", 11, QFont.Bold))
        header.setFixedHeight(36)
        header.setStyleSheet(
            "background-color: #2d2d30; color: #dcdcdc; border-bottom: 1px solid #3e3e42;"
        )
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._content)
        layout.addWidget(scroll)

    def _show_placeholder(self):
        self._clear_content()
        lbl = QLabel("No file loaded.\n\nUse File → Open to load\na .snirf file.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #808080; padding: 20px;")
        self._content_layout.addWidget(lbl)

    def _clear_content(self):
        while self._content_layout.count():
            child = self._content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_info(self, data: SNIRFData):
        """Populate the panel with data from a loaded SNIRF file."""
        self._clear_content()
        import os

        file_group = self._make_group("File")
        form = QFormLayout()
        form.addRow("Name:", self._val_label(os.path.basename(data.filepath)))
        form.addRow("Path:", self._val_label(data.filepath, wrap=True))
        file_group.setLayout(form)
        self._content_layout.addWidget(file_group)

        rec_group = self._make_group("Recording")
        form = QFormLayout()
        form.addRow("Channels:", self._val_label(str(data.n_channels)))
        form.addRow("Timepoints:", self._val_label(str(data.n_timepoints)))
        form.addRow("Duration:", self._val_label(f"{data.duration_seconds:.1f} s"))
        form.addRow("Sampling Rate:", self._val_label(f"{data.sampling_rate:.2f} Hz"))
        rec_group.setLayout(form)
        self._content_layout.addWidget(rec_group)

        probe_group = self._make_group("Probe")
        form = QFormLayout()
        form.addRow("Sources:", self._val_label(str(data.n_sources)))
        form.addRow("Detectors:", self._val_label(str(data.n_detectors)))
        wl_str = ", ".join(f"{w:.0f} nm" for w in data.wavelength_list)
        form.addRow("Wavelengths:", self._val_label(wl_str))
        probe_group.setLayout(form)
        self._content_layout.addWidget(probe_group)

        if data.stimuli:
            stim_group = self._make_group("Stimuli")
            form = QFormLayout()
            for stim in data.stimuli:
                form.addRow(stim.name + ":", self._val_label(f"{len(stim.onset)} events"))
            stim_group.setLayout(form)
            self._content_layout.addWidget(stim_group)

        if data.metadata:
            meta_group = self._make_group("Metadata")
            form = QFormLayout()
            for k, v in list(data.metadata.items())[:10]:
                form.addRow(k + ":", self._val_label(str(v)))
            meta_group.setLayout(form)
            self._content_layout.addWidget(meta_group)

        self._content_layout.addStretch()

    def clear_info(self):
        self._show_placeholder()

    @staticmethod
    def _make_group(title: str) -> QGroupBox:
        grp = QGroupBox(title)
        grp.setStyleSheet("""
            QGroupBox {
                font-weight: bold; color: #61afef;
                border: 1px solid #3e3e42; border-radius: 4px;
                margin-top: 10px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """)
        return grp

    @staticmethod
    def _val_label(text: str, wrap: bool = False) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #dcdcdc;")
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if wrap:
            lbl.setWordWrap(True)
        return lbl
