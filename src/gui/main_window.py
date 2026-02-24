"""
MainWindow — Primary application window for LeafNIRS.

Assembles the file-info panel, processing panel, graph widget,
status bar, and menus into a dark-themed professional layout.
"""
from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QAction, QFileDialog, QMessageBox, QStatusBar, QLabel,
    QMenuBar, QSplitter, QFrame, QScrollArea,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon

from core.data_manager import DataManager
from gui.file_info_panel import FileInfoPanel
from gui.graph_widget import GraphWidget
from gui.processing_panel import ProcessingPanel
from processing.pipeline import ProcessingPipeline, PipelineState


# ═══════════════════════════════════════════════
#  Dark Theme Stylesheet
# ═══════════════════════════════════════════════
_DARK_STYLE = """
/* ── Global ────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #dcdcdc;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 13px;
}

/* ── Menu bar ──────────────────────────── */
QMenuBar {
    background-color: #2d2d30;
    color: #dcdcdc;
    border-bottom: 1px solid #3e3e42;
    padding: 2px;
}
QMenuBar::item:selected {
    background-color: #3e3e42;
    border-radius: 3px;
}
QMenu {
    background-color: #2d2d30;
    color: #dcdcdc;
    border: 1px solid #3e3e42;
}
QMenu::item:selected {
    background-color: #094771;
}

/* ── Status bar ────────────────────────── */
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    font-size: 12px;
    border-top: none;
}
QStatusBar QLabel {
    color: #ffffff;
    padding: 0 8px;
}

/* ── Scroll bars ───────────────────────── */
QScrollBar:vertical {
    background: #1e1e1e;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #3e3e42;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #505054;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #1e1e1e;
    height: 10px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #3e3e42;
    min-width: 30px;
    border-radius: 5px;
}

/* ── Splitter ──────────────────────────── */
QSplitter::handle {
    background-color: #3e3e42;
}
QSplitter::handle:horizontal { width: 2px; }

/* ── Tooltips ──────────────────────────── */
QToolTip {
    background-color: #2d2d30;
    color: #dcdcdc;
    border: 1px solid #3e3e42;
    padding: 4px;
}

/* ── Checkboxes ────────────────────────── */
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #555;
    border-radius: 3px;
    background: #2d2d30;
}
QCheckBox::indicator:checked {
    background: #007acc;
    border-color: #007acc;
}

/* ── Group boxes ───────────────────────── */
QGroupBox {
    font-weight: bold;
    color: #61afef;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
}
"""


class MainWindow(QMainWindow):
    """LeafNIRS main application window."""

    def __init__(self):
        super().__init__()
        self._data_manager = DataManager(self)
        self._pipeline = None       # ProcessingPipeline, set on data load
        self._current_data = None   # SNIRFData from loader
        self._build_ui()
        self._connect_signals()
        self._update_status("Ready — open a .snirf file to begin")

    # ══════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════

    def _build_ui(self):
        self.setWindowTitle("LeafNIRS — fNIRS Brain Mapping Tool")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1400, 850)
        self.setStyleSheet(_DARK_STYLE)

        # ── Menu bar ──
        self._create_menus()

        # ── Central widget ──
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter: left panel | graph
        splitter = QSplitter(Qt.Horizontal)

        # Left sidebar: file info + processing controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._file_info = FileInfoPanel()
        left_layout.addWidget(self._file_info)

        self._processing_panel = ProcessingPanel()
        scroll = QScrollArea()
        scroll.setWidget(self._processing_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        left_layout.addWidget(scroll)

        self._graph = GraphWidget()

        splitter.addWidget(left_panel)
        splitter.addWidget(self._graph)
        splitter.setStretchFactor(0, 0)   # left panel fixed
        splitter.setStretchFactor(1, 1)   # graph stretches
        splitter.setSizes([280, 1120])

        layout.addWidget(splitter)

        # ── Status bar ──
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Ready")
        self._file_label = QLabel("")
        self._channels_label = QLabel("")
        self._status_bar.addWidget(self._status_label, stretch=1)
        self._status_bar.addPermanentWidget(self._file_label)
        self._status_bar.addPermanentWidget(self._channels_label)

    def _create_menus(self):
        menu_bar = self.menuBar()

        # ── File menu ──
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open SNIRF…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a .snirf file for analysis")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        close_action = QAction("&Close File", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self._on_close_file)
        file_menu.addAction(close_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ── View menu ──
        view_menu = menu_bar.addMenu("&View")

        # Loader submenu
        loader_menu = view_menu.addMenu("SNIRF &Loader")
        self._loader_lib_action = QAction("snirf library (Method A)", self, checkable=True)
        self._loader_h5py_action = QAction("h5py raw (Method B)", self, checkable=True)
        self._loader_h5py_action.setChecked(True)
        self._loader_lib_action.triggered.connect(lambda: self._switch_loader("snirf-library"))
        self._loader_h5py_action.triggered.connect(lambda: self._switch_loader("h5py-raw"))
        loader_menu.addAction(self._loader_lib_action)
        loader_menu.addAction(self._loader_h5py_action)

        # ── Help menu ──
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About LeafNIRS", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ══════════════════════════════════════════
    #  Signal Connections
    # ══════════════════════════════════════════

    def _connect_signals(self):
        self._data_manager.data_loaded.connect(self._on_data_loaded)
        self._data_manager.data_cleared.connect(self._on_data_cleared)
        self._data_manager.error_occurred.connect(self._on_error)

        # Processing panel signals
        self._processing_panel.convert_od_clicked.connect(self._on_convert_od)
        self._processing_panel.apply_filter_clicked.connect(self._on_apply_filter)
        self._processing_panel.reset_clicked.connect(self._on_reset_processing)
        self._processing_panel.view_raw_clicked.connect(
            lambda: self._on_switch_view(PipelineState.RAW))
        self._processing_panel.view_od_clicked.connect(
            lambda: self._on_switch_view(PipelineState.OD))
        self._processing_panel.view_filtered_clicked.connect(
            lambda: self._on_switch_view(PipelineState.FILTERED))

    # ══════════════════════════════════════════
    #  Slots / Handlers
    # ══════════════════════════════════════════

    def _on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open SNIRF File",
            "",
            "SNIRF Files (*.snirf);;All Files (*)",
        )
        if filepath:
            self._update_status(f"Loading {os.path.basename(filepath)}…")
            self._data_manager.load_file(filepath)

    def _on_close_file(self):
        self._data_manager.clear()

    def _on_data_loaded(self, data):
        self._current_data = data
        self._pipeline = ProcessingPipeline(data.intensity, data.sampling_rate)
        self._file_info.update_info(data)
        self._graph.plot_data(data)
        self._processing_panel.set_enabled(True)
        fname = os.path.basename(data.filepath)

        # Auto-apply processing if in Auto mode
        if self._processing_panel.is_auto:
            try:
                low = self._processing_panel.filter_low
                high = self._processing_panel.filter_high
                order = self._processing_panel.filter_order
                self._pipeline.apply_bandpass(low=low, high=high, order=order)
                self._graph.update_data(
                    self._pipeline.result.active_data,
                    self._current_data,
                    self._pipeline.result.state_label,
                )
                self._update_status(
                    f"Loaded: {fname} — auto-processed (OD + bandpass {low}–{high} Hz)"
                )
            except Exception as e:
                # Fall back to raw if auto-processing fails
                self._update_status(f"Loaded: {fname} — auto-processing failed: {e}")
        else:
            self._update_status(f"Loaded: {fname}")

        self._sync_processing_state()
        self._file_label.setText(f"\U0001f4c1 {fname}")
        self._channels_label.setText(
            f"\U0001f4ca {data.n_channels} ch  |  "
            f"\u23f1 {data.duration_seconds:.1f}s  |  "
            f"\U0001f4c8 {data.sampling_rate:.1f} Hz"
        )

    def _on_data_cleared(self):
        self._current_data = None
        self._pipeline = None
        self._file_info.clear_info()
        self._graph.clear_plot()
        self._processing_panel.set_enabled(False)
        self._update_status("Ready — open a .snirf file to begin")
        self._file_label.setText("")
        self._channels_label.setText("")

    def _on_error(self, message: str):
        self._update_status("Error loading file")
        QMessageBox.critical(
            self,
            "Load Error",
            f"Failed to load SNIRF file:\n\n{message}",
        )

    def _switch_loader(self, loader_name: str):
        self._data_manager.use_loader(loader_name)
        is_lib = loader_name == "snirf-library"
        self._loader_lib_action.setChecked(is_lib)
        self._loader_h5py_action.setChecked(not is_lib)
        self._update_status(f"Loader switched to: {loader_name}")

    def _on_about(self):
        QMessageBox.about(
            self,
            "About LeafNIRS",
            "<h2>LeafNIRS v0.1.0</h2>"
            "<p>A Python-based fNIRS Brain Mapping Tool for<br>"
            "Signal Processing and Visualization.</p>"
            "<p><b>Phase 1:</b> Data Loading & Basic Visualization</p>"
            "<hr>"
            "<p>Developers:<br>"
            "Ali Umut Sezgin · Arda Telci · Özgür Efe Zurnaci</p>"
            "<p>Supervisor: Prof. Dr. Ata Akin<br>"
            "Acibadem Mehmet Ali Aydinlar University</p>"
            "<hr>"
            "<p>Open source · MIT License · Python 3.12</p>"
        )

    # ══════════════════════════════════════════
    #  Processing Handlers
    # ══════════════════════════════════════════

    def _on_convert_od(self):
        if not self._pipeline:
            return
        self._update_status("Converting to optical density...")
        try:
            self._pipeline.convert_to_od()
            self._graph.update_data(
                self._pipeline.result.active_data,
                self._current_data,
                self._pipeline.result.state_label,
            )
            self._sync_processing_state()
            self._update_status("Converted to optical density")
        except Exception as e:
            self._on_error(f"OD conversion failed: {e}")

    def _on_apply_filter(self, low: float, high: float, order: int):
        if not self._pipeline:
            return
        self._update_status(f"Applying bandpass filter ({low}–{high} Hz)...")
        try:
            self._pipeline.apply_bandpass(low=low, high=high, order=order)
            self._graph.update_data(
                self._pipeline.result.active_data,
                self._current_data,
                self._pipeline.result.state_label,
            )
            self._sync_processing_state()
            self._update_status(
                f"Bandpass filtered: {low}–{high} Hz, order {order}"
            )
        except Exception as e:
            self._on_error(f"Filter failed: {e}")

    def _on_reset_processing(self):
        if not self._pipeline:
            return
        self._pipeline.reset()
        self._graph.plot_data(self._current_data)
        self._sync_processing_state()
        self._update_status("Reset to raw intensity")

    def _on_switch_view(self, state: PipelineState):
        if not self._pipeline:
            return
        try:
            self._pipeline.set_view(state)
            self._graph.update_data(
                self._pipeline.result.active_data,
                self._current_data,
                self._pipeline.result.state_label,
            )
            self._sync_processing_state()
        except ValueError:
            pass  # Data not available for this view

    def _sync_processing_state(self):
        if self._pipeline:
            r = self._pipeline.result
            self._processing_panel.update_state(
                r.state_label,
                has_od=r.od is not None,
                has_filtered=r.filtered is not None,
            )

    # ── Helpers ───────────────────────────────

    def _update_status(self, text: str):
        self._status_label.setText(f"  {text}")
