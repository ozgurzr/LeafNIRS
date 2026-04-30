"""MainWindow — Primary application window for LeafNIRS."""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QAction, QFileDialog, QMessageBox, QStatusBar, QLabel,
    QMenuBar, QSplitter, QFrame, QScrollArea, QTabWidget,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon

from core.data_manager import DataManager
from gui.file_info_panel import FileInfoPanel
from gui.graph_widget import GraphWidget
from gui.processing_panel import ProcessingPanel
from gui.epoch_viewer import EpochViewer
from gui.probe_map_widget import ProbeMapWidget
from gui.brain_viewer_widget import BrainViewerWidget
from processing.pipeline import ProcessingPipeline, PipelineState
from processing.epoch_extraction import compute_condition_average
from processing.export_manager import export_csv, export_matlab


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
QMenu::item:selected {
    background-color: #094771;
}
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
QSplitter::handle {
    background-color: #3e3e42;
}
QSplitter::handle:horizontal { width: 2px; }
QToolTip {
    background-color: #2d2d30;
    color: #dcdcdc;
    border: 1px solid #3e3e42;
    padding: 4px;
}
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
        self._pipeline = None
        self._current_data = None
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
        self._epoch_viewer = EpochViewer()
        self._epoch_viewer.setMaximumHeight(280)
        self._probe_map = ProbeMapWidget()
        self._brain_viewer = BrainViewerWidget()

        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(self._graph)
        v_splitter.addWidget(self._epoch_viewer)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)
        v_splitter.setSizes([600, 250])

        self._glm_tabs = QTabWidget()
        self._glm_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3e3e42; background: #1e1e1e; }
            QTabBar::tab {
                background: #2d2d30; color: #abb2bf; padding: 6px 14px;
                border: 1px solid #3e3e42; border-bottom: none;
                margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #094771; color: #fff; }
            QTabBar::tab:hover { background: #3e3e42; }
        """)
        self._glm_tabs.addTab(self._probe_map, "📍 2D Probe Map")
        self._glm_tabs.addTab(self._brain_viewer, "🧠 3D Brain")
        self._glm_tabs.setVisible(False)

        h_right_splitter = QSplitter(Qt.Horizontal)
        h_right_splitter.addWidget(v_splitter)
        h_right_splitter.addWidget(self._glm_tabs)
        h_right_splitter.setStretchFactor(0, 3)
        h_right_splitter.setStretchFactor(1, 1)
        h_right_splitter.setSizes([900, 0])
        self._h_right_splitter = h_right_splitter

        splitter.addWidget(left_panel)
        splitter.addWidget(h_right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1120])

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

        self._export_csv_action = QAction("Export as &CSV…", self)
        self._export_csv_action.setShortcut("Ctrl+Shift+C")
        self._export_csv_action.setStatusTip("Export current data to CSV")
        self._export_csv_action.setEnabled(False)
        self._export_csv_action.triggered.connect(self._on_export_csv)
        file_menu.addAction(self._export_csv_action)

        self._export_mat_action = QAction("Export as &MATLAB…", self)
        self._export_mat_action.setShortcut("Ctrl+Shift+M")
        self._export_mat_action.setStatusTip("Export current data to .mat file")
        self._export_mat_action.setEnabled(False)
        self._export_mat_action.triggered.connect(self._on_export_matlab)
        file_menu.addAction(self._export_mat_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menu_bar.addMenu("&View")

        loader_menu = view_menu.addMenu("SNIRF &Loader")
        self._loader_lib_action = QAction("snirf library (Method A)", self, checkable=True)
        self._loader_h5py_action = QAction("h5py raw (Method B)", self, checkable=True)
        self._loader_lib_action.setChecked(True)
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

        self._processing_panel.convert_od_clicked.connect(self._on_convert_od)
        self._processing_panel.apply_correction_clicked.connect(self._on_apply_correction)
        self._processing_panel.apply_filter_clicked.connect(self._on_apply_filter)
        self._processing_panel.convert_conc_clicked.connect(self._on_convert_concentration)
        self._processing_panel.apply_all_clicked.connect(self._on_apply_all)
        self._processing_panel.reset_clicked.connect(self._on_reset_processing)
        self._processing_panel.view_raw_clicked.connect(
            lambda: self._on_switch_view(PipelineState.RAW))
        self._processing_panel.view_od_clicked.connect(
            lambda: self._on_switch_view(PipelineState.OD))
        self._processing_panel.view_corrected_clicked.connect(
            lambda: self._on_switch_view(PipelineState.CORRECTED))
        self._processing_panel.view_filtered_clicked.connect(
            lambda: self._on_switch_view(PipelineState.FILTERED))
        self._processing_panel.view_conc_clicked.connect(
            lambda: self._on_switch_view(PipelineState.CONCENTRATION))

        self._epoch_viewer.compute_requested.connect(self._on_compute_epochs)
        self._processing_panel.run_glm_clicked.connect(self._on_run_glm)

    def _on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open SNIRF File", "",
            "SNIRF Files (*.snirf);;All Files (*)",
        )
        if filepath:
            self._update_status(f"Loading {os.path.basename(filepath)}…")
            self._data_manager.load_file(filepath)

    def _on_close_file(self):
        self._data_manager.clear()

    def _on_data_loaded(self, data):
        self._current_data = data
        self._pipeline = ProcessingPipeline(
            data.intensity, data.sampling_rate,
            channels=data.channels, probe=data.probe,
        )
        self._file_info.update_info(data)
        self._graph.plot_data(data)
        self._processing_panel.set_enabled(True)
        fname = os.path.basename(data.filepath)

        if self._processing_panel.is_auto:
            try:
                low = self._processing_panel.filter_low
                high = self._processing_panel.filter_high
                order = self._processing_panel.filter_order
                self._pipeline.apply_motion_correction(method='tddr')
                self._pipeline.apply_bandpass(low=low, high=high, order=order)
                self._pipeline.convert_to_concentration()
                r = self._pipeline.result
                self._graph.plot_concentration(
                    r.hbo, r.hbr, self._current_data, r.pair_labels,
                )
                self._update_status(
                    f"Loaded: {fname} — auto-processed (OD + TDDR + filter + HbO/HbR)"
                )
            except Exception as e:
                import traceback; traceback.print_exc()
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

        if data.stimuli:
            self._epoch_viewer.set_conditions([s.name for s in data.stimuli])
        else:
            self._epoch_viewer.set_conditions([])

    def _on_data_cleared(self):
        self._current_data = None
        self._pipeline = None
        self._file_info.clear_info()
        self._graph.clear_plot()
        self._epoch_viewer.clear()
        self._probe_map.clear()
        self._brain_viewer.clear()
        self._glm_tabs.setVisible(False)
        self._processing_panel.set_enabled(False)
        self._update_status("Ready — open a .snirf file to begin")
        self._file_label.setText("")
        self._channels_label.setText("")

    def _on_error(self, message: str):
        self._update_status("Error loading file")
        QMessageBox.critical(
            self, "Load Error",
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

    def _on_apply_all(self):
        """Run full pipeline: OD → TDDR → bandpass → MBLL."""
        if not self._pipeline:
            return
        self._update_status("Applying full pipeline...")
        try:
            low = self._processing_panel.filter_low
            high = self._processing_panel.filter_high
            order = self._processing_panel.filter_order
            self._pipeline.apply_motion_correction(method='tddr')
            self._pipeline.apply_bandpass(low=low, high=high, order=order)
            self._pipeline.convert_to_concentration()
            r = self._pipeline.result
            self._graph.plot_concentration(
                r.hbo, r.hbr, self._current_data, r.pair_labels,
            )
            self._sync_processing_state()
            self._update_status(
                f"Full pipeline applied: {len(r.pair_labels)} pairs → HbO/HbR"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._on_error(f"Pipeline failed: {e}")

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

    def _on_apply_correction(self, method: str):
        if not self._pipeline:
            return
        self._update_status(f"Applying motion correction ({method.upper()})...")
        try:
            self._pipeline.apply_motion_correction(method=method)
            r = self._pipeline.result
            n_artifacts = r.artifact_mask.sum() if r.artifact_mask is not None else 0
            self._graph.update_data(r.active_data, self._current_data, r.state_label)
            self._sync_processing_state()
            self._update_status(
                f"Motion corrected ({method.upper()}): {n_artifacts} artifact samples"
            )
        except Exception as e:
            self._on_error(f"Motion correction failed: {e}")

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
            self._update_status(f"Bandpass filtered: {low}–{high} Hz, order {order}")
        except Exception as e:
            self._on_error(f"Filter failed: {e}")

    def _on_convert_concentration(self):
        if not self._pipeline:
            return
        self._update_status("Converting to HbO/HbR concentrations (MBLL)...")
        try:
            self._pipeline.convert_to_concentration()
            r = self._pipeline.result
            self._graph.plot_concentration(
                r.hbo, r.hbr, self._current_data, r.pair_labels,
            )
            self._sync_processing_state()
            self._update_status(f"MBLL: {len(r.pair_labels)} S-D pairs → HbO/HbR")
        except Exception as e:
            self._on_error(f"MBLL conversion failed: {e}")

    def _on_reset_processing(self):
        if not self._pipeline:
            return
        self._pipeline.reset()
        self._graph.plot_data(self._current_data)
        self._probe_map.clear()
        self._brain_viewer.clear()
        self._glm_tabs.setVisible(False)
        self._sync_processing_state()
        self._update_status("Reset to raw intensity")

    def _on_switch_view(self, state: PipelineState):
        if not self._pipeline:
            return
        try:
            self._pipeline.set_view(state)
            r = self._pipeline.result
            if state == PipelineState.CONCENTRATION:
                self._graph.plot_concentration(
                    r.hbo, r.hbr, self._current_data, r.pair_labels,
                )
            else:
                self._graph.update_data(r.active_data, self._current_data, r.state_label)
            self._sync_processing_state()
        except ValueError:
            pass

    def _sync_processing_state(self):
        if self._pipeline:
            r = self._pipeline.result
            self._processing_panel.update_state(
                r.state_label,
                has_od=r.od is not None,
                has_filtered=r.filtered is not None,
                has_corrected=r.corrected is not None,
                has_concentration=r.hbo is not None,
            )
            has_data = self._current_data is not None
            self._export_csv_action.setEnabled(has_data)
            self._export_mat_action.setEnabled(has_data)

    def _update_status(self, text: str):
        self._status_label.setText(f"  {text}")

    def _on_compute_epochs(self, condition: str, pre_sec: float, post_sec: float):
        """Compute block average for the requested condition."""
        if not self._pipeline or not self._current_data:
            return

        r = self._pipeline.result
        if r.hbo is None or r.hbr is None:
            self._on_error(
                "HbO/HbR data required for block averaging.\n"
                "Run the full pipeline first (Apply All)."
            )
            return

        stim = None
        for s in self._current_data.stimuli:
            if s.name == condition:
                stim = s
                break

        if stim is None:
            self._on_error(f"Stimulus condition '{condition}' not found.")
            return

        self._update_status(f"Computing block average for '{condition}'...")
        try:
            result = compute_condition_average(
                hbo=r.hbo, hbr=r.hbr,
                time=self._current_data.time,
                onsets=stim.onset,
                condition_name=condition,
                pair_labels=r.pair_labels,
                pre_sec=pre_sec, post_sec=post_sec,
            )
            self._epoch_viewer.set_results([result])
            self._update_status(
                f"Block average: '{condition}' — {result.n_trials} trials averaged"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._on_error(f"Block averaging failed: {e}")

    def _on_run_glm(self):
        """Run GLM statistical analysis on concentration data."""
        if not self._pipeline or not self._current_data:
            return

        r = self._pipeline.result
        if r.hbo is None or r.hbr is None:
            self._on_error(
                "HbO/HbR data required for GLM.\n"
                "Run the full pipeline first (Apply All)."
            )
            return

        if not self._current_data.stimuli:
            self._on_error("No stimulus conditions found in this file.")
            return

        self._update_status("Running GLM analysis...")
        try:
            glm_hbo, glm_hbr = self._pipeline.run_glm(self._current_data.stimuli)
            n_cond = len(glm_hbo.contrast_names)
            n_pairs = len(glm_hbo.pair_labels)
            n_sig = int(np.sum(glm_hbo.p_value < 0.05))
            self._probe_map.set_glm_results(
                glm_hbo, glm_hbr, self._current_data.probe, self._current_data.channels,
            )
            self._brain_viewer.set_glm_results(
                glm_hbo, glm_hbr, self._current_data.probe, self._current_data.channels,
            )
            self._glm_tabs.setVisible(True)
            self._h_right_splitter.setSizes([700, 400])
            self._sync_processing_state()
            self._update_status(
                f"GLM: {n_cond} conditions × {n_pairs} pairs — "
                f"{n_sig} significant (p<0.05)"
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._on_error(f"GLM analysis failed: {e}")

    def _on_export_csv(self):
        """Export current pipeline view to CSV."""
        if not self._pipeline or not self._current_data:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export as CSV", "", "CSV Files (*.csv);;All Files (*)",
        )
        if not filepath:
            return
        try:
            r = self._pipeline.result
            is_conc = r.state == PipelineState.CONCENTRATION and r.hbo is not None
            files = export_csv(
                filepath=filepath,
                time=self._current_data.time,
                data=r.hbo if is_conc else r.active_data,
                data_label=r.state_label,
                pair_labels=r.pair_labels if is_conc else None,
                channels=self._current_data.channels if not is_conc else None,
                hbr=r.hbr if is_conc else None,
                glm_hbo=r.glm_hbo,
                glm_hbr=r.glm_hbr,
            )
            self._update_status(f"Exported {len(files)} CSV file(s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._on_error(f"CSV export failed: {e}")

    def _on_export_matlab(self):
        """Export current pipeline view to MATLAB .mat."""
        if not self._pipeline or not self._current_data:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export as MATLAB", "", "MATLAB Files (*.mat);;All Files (*)",
        )
        if not filepath:
            return
        try:
            r = self._pipeline.result
            is_conc = r.state == PipelineState.CONCENTRATION and r.hbo is not None
            export_matlab(
                filepath=filepath,
                time=self._current_data.time,
                data=r.hbo if is_conc else r.active_data,
                data_label=r.state_label,
                sampling_rate=self._current_data.sampling_rate,
                pair_labels=r.pair_labels if is_conc else None,
                channels=self._current_data.channels if not is_conc else None,
                hbr=r.hbr if is_conc else None,
                probe=self._current_data.probe,
                glm_hbo=r.glm_hbo,
                glm_hbr=r.glm_hbr,
            )
            self._update_status(f"Exported MATLAB file: {os.path.basename(filepath)}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._on_error(f"MATLAB export failed: {e}")
