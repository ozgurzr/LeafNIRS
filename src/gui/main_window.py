"""MainWindow — Primary application window for LeafNIRS."""
from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QAction, QFileDialog, QMessageBox, QStatusBar, QLabel,
    QSplitter,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont

from core.data_manager import DataManager
from gui.file_info_panel import FileInfoPanel
from gui.graph_widget import GraphWidget

_DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #dcdcdc;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 13px;
}
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
QMenu::item:selected { background-color: #094771; }
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    font-size: 12px;
    border-top: none;
}
QStatusBar QLabel { color: #ffffff; padding: 0 8px; }
QScrollBar:vertical {
    background: #1e1e1e; width: 10px; border: none;
}
QScrollBar::handle:vertical {
    background: #3e3e42; min-height: 30px; border-radius: 5px;
}
QScrollBar::handle:vertical:hover { background: #505054; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #1e1e1e; height: 10px; border: none;
}
QScrollBar::handle:horizontal {
    background: #3e3e42; min-width: 30px; border-radius: 5px;
}
QSplitter::handle { background-color: #3e3e42; }
QSplitter::handle:horizontal { width: 2px; }
QToolTip {
    background-color: #2d2d30; color: #dcdcdc;
    border: 1px solid #3e3e42; padding: 4px;
}
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #555; border-radius: 3px;
    background: #2d2d30;
}
QCheckBox::indicator:checked { background: #007acc; border-color: #007acc; }
QGroupBox {
    font-weight: bold; color: #61afef;
    border: 1px solid #3e3e42; border-radius: 4px;
    margin-top: 10px; padding-top: 14px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
"""


class MainWindow(QMainWindow):
    """LeafNIRS main application window."""

    def __init__(self):
        super().__init__()
        self._data_manager = DataManager(self)
        self._build_ui()
        self._connect_signals()
        self._update_status("Ready — open a .snirf file to begin")

    def _build_ui(self):
        self.setWindowTitle("LeafNIRS — fNIRS Brain Mapping Tool")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1400, 850)
        self.setStyleSheet(_DARK_STYLE)

        self._create_menus()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        self._file_info = FileInfoPanel()
        self._graph = GraphWidget()
        splitter.addWidget(self._file_info)
        splitter.addWidget(self._graph)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

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

        view_menu = menu_bar.addMenu("&View")
        loader_menu = view_menu.addMenu("SNIRF &Loader")
        self._loader_lib_action = QAction("[Alternative] snirf library", self, checkable=True)
        self._loader_h5py_action = QAction("h5py (Default)", self, checkable=True)
        self._loader_h5py_action.setChecked(True)
        self._loader_lib_action.triggered.connect(lambda: self._switch_loader("snirf-library"))
        self._loader_h5py_action.triggered.connect(lambda: self._switch_loader("h5py-raw"))
        loader_menu.addAction(self._loader_lib_action)
        loader_menu.addAction(self._loader_h5py_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About LeafNIRS", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        self._data_manager.data_loaded.connect(self._on_data_loaded)
        self._data_manager.data_cleared.connect(self._on_data_cleared)
        self._data_manager.error_occurred.connect(self._on_error)

    def _on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open SNIRF File", "",
            "SNIRF Files (*.snirf);;All Files (*)",
        )
        if filepath:
            self._update_status(f"Loading {os.path.basename(filepath)}…")
            self._data_manager.load_file(filepath)

    def _on_close_file(self):
        self._data_manager.close_file()

    def _on_data_loaded(self, data):
        self._file_info.update_info(data)
        self._graph.plot_data(data)
        fname = os.path.basename(data.filepath)
        self._update_status(f"Loaded: {fname}")
        self._file_label.setText(f"📁 {fname}")
        self._channels_label.setText(
            f"📊 {data.n_channels} ch  |  ⏱ {data.duration_seconds:.1f}s  |  📈 {data.sampling_rate:.1f} Hz"
        )

    def _on_data_cleared(self):
        self._file_info.clear_info()
        self._graph.clear_plot()
        self._update_status("Ready — open a .snirf file to begin")
        self._file_label.setText("")
        self._channels_label.setText("")

    def _on_error(self, message: str):
        self._update_status("Error loading file")
        QMessageBox.critical(self, "Load Error", f"Failed to load SNIRF file:\n\n{message}")

    def _switch_loader(self, loader_name: str):
        self._data_manager.set_loader(loader_name)
        is_lib = loader_name == "snirf-library"
        self._loader_lib_action.setChecked(is_lib)
        self._loader_h5py_action.setChecked(not is_lib)
        self._update_status(f"Loader switched to: {loader_name}")

    def _on_about(self):
        QMessageBox.about(
            self, "About LeafNIRS",
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

    def _update_status(self, text: str):
        self._status_label.setText(f"  {text}")
