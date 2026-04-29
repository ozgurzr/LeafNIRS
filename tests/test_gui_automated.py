"""Automated GUI test — drives LeafNIRS through every workflow, saves screenshots."""
import sys
import os
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

SNIRF = r"C:\Users\90546\Desktop\LeafNIRS Plan\data\DATA_MotorNeuron\SUBJID_6082\run1.snirf"
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_screenshots')
LOG_PATH = os.path.join(OUT_DIR, 'test_log.txt')

os.makedirs(OUT_DIR, exist_ok=True)


def _count_checked(gw):
    count = 0
    for pw in gw._pair_widgets:
        if hasattr(pw, 'pair_cb'):
            if pw.pair_cb.isChecked():
                count += 1
        elif hasattr(pw, 'is_checked'):
            if pw.is_checked():
                count += 1
    return count


def _has_curve_data(gw):
    for c in gw._curves.values():
        if c.xData is not None and len(c.xData) > 0:
            return True
    return False


class TestRunner:
    def __init__(self, window, app):
        self.w = window
        self.app = app
        self.step = 0
        self.log_lines = []
        self.errors = []

    def log(self, msg):
        line = f"[{self.step:03d}] {msg}"
        self.log_lines.append(line)
        print(line)

    def grab(self, name, widget=None):
        self.app.processEvents()
        target = widget if widget else self.w
        fname = f"{self.step:03d}_{name}.png"
        path = os.path.join(OUT_DIR, fname)
        target.grab().save(path)
        self.log(f"  IMG: {fname}")
        return fname

    def grab_graph(self, name):
        return self.grab(f"graph_{name}", self.w._graph)

    def grab_probe(self, name):
        if self.w._probe_map.isVisible():
            return self.grab(f"probe2d_{name}", self.w._probe_map)
        return None

    def grab_brain(self, name):
        if self.w._brain_viewer.isVisible():
            return self.grab(f"brain3d_{name}", self.w._brain_viewer)
        return None

    def wait(self, ms=200):
        end = time.time() + ms / 1000.0
        while time.time() < end:
            self.app.processEvents()

    def S(self, desc):
        self.step += 1
        self.log(desc)

    def ok(self, cond, msg):
        if not cond:
            self.log(f"  FAIL: {msg}")
            self.errors.append(f"Step {self.step}: {msg}")
        else:
            self.log(f"  OK: {msg}")

    def run_all(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log(f"={'='*59}")
        self.log(f"LeafNIRS Automated GUI Test — {ts}")
        self.log(f"={'='*59}")

        try:
            self._test_initial()
            self._test_manual_load()
            self._test_raw_channels()
            self._test_raw_views()
            self._test_manual_pipeline()
            self._test_conc_channels()
            self._test_conc_views()
            self._test_glm()
            self._test_glm_controls()
            self._test_reset()
            self._test_apply_all()
            self._test_apply_all_channels()
            self._test_glm_after_apply_all()
            self._test_close_reopen()
            self._test_auto_mode()
        except Exception as e:
            import traceback
            self.log(f"  EXCEPTION: {e}")
            self.log(traceback.format_exc())
            self.errors.append(f"Step {self.step}: {e}")

        self.log("")
        self.log(f"={'='*59}")
        self.log(f"RESULTS: {self.step} steps, {len(self.errors)} failures")
        for e in self.errors:
            self.log(f"  X {e}")
        if not self.errors:
            self.log("  ALL PASSED")
        self.log(f"={'='*59}")

        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))
        print(f"\nLog: {LOG_PATH}")
        print(f"Screenshots: {OUT_DIR}")
        self.app.quit()

    def _test_initial(self):
        self.S("INITIAL STATE")
        self.wait(300)
        self.grab("initial")
        self.ok(self.w._current_data is None, "No data")
        self.ok(not self.w._processing_panel._btn_apply_all.isEnabled(), "Apply All disabled")

    def _test_manual_load(self):
        self.S("MANUAL MODE + LOAD")
        self.w._processing_panel._on_manual_clicked()
        self.wait(100)
        self.w._data_manager.load_file(SNIRF)
        self.wait(500)
        self.grab("loaded_manual")
        self.grab_graph("raw_loaded")
        self.ok(self.w._current_data is not None, "Data loaded")
        self.ok(self.w._current_data.n_channels == 120, "120 ch")
        self.ok(not getattr(self.w._graph, '_conc_mode', False), "Raw mode")

    def _test_raw_channels(self):
        gw = self.w._graph

        self.S("RAW — All")
        gw._select_all()
        self.wait(200)
        self.grab_graph("raw_all")
        self.ok(_count_checked(gw) == len(gw._pair_widgets), "All checked")
        self.ok(_has_curve_data(gw), "Has data")

        self.S("RAW — None")
        gw._select_none()
        self.wait(200)
        self.grab_graph("raw_none")
        self.ok(_count_checked(gw) == 0, "None checked")
        self.ok(not _has_curve_data(gw), "Empty")

        self.S("RAW — First 10")
        gw._spin_first.setValue(10)
        gw._select_first_n()
        self.wait(200)
        self.grab_graph("raw_first10")
        self.ok(_count_checked(gw) == 10, "10 checked")

        self.S("RAW — First 60")
        gw._spin_first.setValue(60)
        gw._select_first_n()
        self.wait(200)
        self.grab_graph("raw_first60")
        self.ok(_has_curve_data(gw), "Has data")

        self.S("RAW — WL1 only")
        gw._spin_first.setValue(5)
        gw._select_first_n()
        gw._set_wl_filter(1)
        self.wait(200)
        self.grab_graph("raw_wl1")

        self.S("RAW — WL2 only")
        gw._set_wl_filter(2)
        self.wait(200)
        self.grab_graph("raw_wl2")

        self.S("RAW — Both WL")
        gw._set_wl_filter(None)
        self.wait(200)
        self.grab_graph("raw_wl_both")

    def _test_raw_views(self):
        self.S("RAW — Stacked")
        self.w._graph._set_view_mode(True)
        self.wait(200)
        self.grab_graph("raw_stacked")
        self.ok(self.w._graph._stacked, "Stacked on")
        self.ok(_has_curve_data(self.w._graph), "Has data")

        self.S("RAW — Overlaid")
        self.w._graph._set_view_mode(False)
        self.wait(200)
        self.grab_graph("raw_overlaid")
        self.ok(not self.w._graph._stacked, "Stacked off")
        self.ok(_has_curve_data(self.w._graph), "Has data")

    def _test_manual_pipeline(self):
        self.S("OD conversion")
        self.w._on_convert_od()
        self.wait(300)
        self.grab_graph("od")
        self.ok(self.w._pipeline.result.od is not None, "OD exists")
        self.ok(_has_curve_data(self.w._graph), "Has data")

        self.S("TDDR correction")
        self.w._on_apply_correction("tddr")
        self.wait(300)
        self.grab_graph("corrected")
        self.ok(self.w._pipeline.result.corrected is not None, "Corrected exists")
        self.ok(_has_curve_data(self.w._graph), "Has data")

        self.S("Bandpass filter")
        self.w._on_apply_filter(0.01, 0.1, 3)
        self.wait(300)
        self.grab_graph("filtered")
        self.ok(self.w._pipeline.result.filtered is not None, "Filtered exists")
        self.ok(_has_curve_data(self.w._graph), "Has data")

        self.S("MBLL (HbO/HbR)")
        self.w._on_convert_concentration()
        self.wait(500)
        self.grab("after_mbll")
        self.grab_graph("conc")
        self.ok(self.w._pipeline.result.hbo is not None, "HbO exists")
        self.ok(self.w._pipeline.result.hbr is not None, "HbR exists")
        self.ok(self.w._pipeline.result.hbo.shape[1] == 60, "60 pairs")
        self.ok(getattr(self.w._graph, '_conc_mode', False), "Conc mode")

    def _test_conc_channels(self):
        gw = self.w._graph

        self.S("CONC — All")
        gw._select_all()
        self.wait(300)
        self.grab_graph("conc_all")
        self.ok(_count_checked(gw) == len(gw._pair_widgets), "All checked")
        self.ok(_has_curve_data(gw), "Not empty")

        self.S("CONC — None")
        gw._select_none()
        self.wait(200)
        self.grab_graph("conc_none")
        self.ok(not _has_curve_data(gw), "Empty")

        self.S("CONC — First 5")
        gw._spin_first.setValue(5)
        gw._select_first_n()
        self.wait(200)
        self.grab_graph("conc_first5")
        self.ok(_has_curve_data(gw), "Has data")

        self.S("CONC — First 30")
        gw._spin_first.setValue(30)
        gw._select_first_n()
        self.wait(200)
        self.grab_graph("conc_first30")

    def _test_conc_views(self):
        gw = self.w._graph

        self.S("CONC — Stacked")
        gw._set_view_mode(True)
        self.wait(300)
        self.grab_graph("conc_stacked")
        self.ok(gw._stacked, "Stacked on")
        self.ok(_has_curve_data(gw), "Not empty")

        self.S("CONC — Overlaid")
        gw._set_view_mode(False)
        self.wait(300)
        self.grab_graph("conc_overlaid")
        self.ok(not gw._stacked, "Stacked off")
        self.ok(_has_curve_data(gw), "Not empty")

    def _test_glm(self):
        self.S("GLM analysis")
        self.w._on_run_glm()
        self.wait(500)
        self.grab("after_glm")
        self.grab_graph("glm_graph")
        self.grab_probe("glm_default")
        self.grab_brain("glm_default")
        self.ok(self.w._glm_tabs.isVisible(), "GLM tabs visible")

    def _test_glm_controls(self):
        # 2D tab
        self.S("GLM 2D — tab select")
        self.w._glm_tabs.setCurrentIndex(0)
        self.wait(200)
        self.grab_probe("tab_2d")

        pm = self.w._probe_map
        if pm._combo_cond.count() > 1:
            self.S("GLM 2D — cond 2")
            pm._combo_cond.setCurrentIndex(1)
            self.wait(200)
            self.grab_probe("cond2")

        self.S("GLM 2D — HbR")
        pm._combo_chrom.setCurrentText("HbR")
        self.wait(200)
        self.grab_probe("hbr")

        self.S("GLM 2D — p<0.01")
        pm._slider_p.setValue(1)
        self.wait(200)
        self.grab_probe("p001")

        self.S("GLM 2D — p<0.10")
        pm._slider_p.setValue(10)
        self.wait(200)
        self.grab_probe("p010")

        pm._combo_chrom.setCurrentText("HbO")
        pm._combo_cond.setCurrentIndex(0)
        pm._slider_p.setValue(5)
        self.wait(100)

        # 3D tab
        self.S("GLM 3D — tab select")
        self.w._glm_tabs.setCurrentIndex(1)
        self.wait(300)
        self.grab_brain("tab_3d")

        bv = self.w._brain_viewer
        if bv._combo_cond.count() > 2:
            self.S("GLM 3D — cond 3")
            bv._combo_cond.setCurrentIndex(2)
            self.wait(200)
            self.grab_brain("cond3")

        self.S("GLM 3D — HbR")
        bv._combo_chrom.setCurrentText("HbR")
        self.wait(200)
        self.grab_brain("hbr")

        bv._combo_chrom.setCurrentText("HbO")
        bv._combo_cond.setCurrentIndex(0)

    def _test_reset(self):
        self.S("RESET to Raw")
        self.w._on_reset_processing()
        self.wait(300)
        self.grab("after_reset")
        self.grab_graph("reset_raw")
        self.ok(not getattr(self.w._graph, '_conc_mode', False), "Not conc mode")
        self.ok(not self.w._glm_tabs.isVisible(), "GLM hidden")
        self.ok(_has_curve_data(self.w._graph), "Has data")

    def _test_apply_all(self):
        self.S("APPLY ALL")
        self.w._on_apply_all()
        self.wait(500)
        self.grab("after_apply_all")
        self.grab_graph("apply_all_conc")
        self.ok(self.w._pipeline.result.hbo is not None, "HbO exists")
        self.ok(getattr(self.w._graph, '_conc_mode', False), "Conc mode")
        self.ok(_has_curve_data(self.w._graph), "Has data")

    def _test_apply_all_channels(self):
        gw = self.w._graph

        self.S("APPLY ALL + All channels")
        gw._select_all()
        self.wait(300)
        self.grab_graph("apply_all_ch_all")
        self.ok(_has_curve_data(gw), "Not empty")

        self.S("APPLY ALL + Stacked all")
        gw._set_view_mode(True)
        self.wait(300)
        self.grab_graph("apply_all_stacked")
        self.ok(_has_curve_data(gw), "Not empty")

        self.S("APPLY ALL + Overlaid 5")
        gw._set_view_mode(False)
        gw._spin_first.setValue(5)
        gw._select_first_n()
        self.wait(200)
        self.grab_graph("apply_all_ov5")

    def _test_glm_after_apply_all(self):
        self.S("GLM after Apply All")
        self.w._on_run_glm()
        self.wait(500)
        self.grab("glm_apply_all")
        self.w._glm_tabs.setCurrentIndex(0)
        self.wait(200)
        self.grab_probe("after_apply_all")
        self.w._glm_tabs.setCurrentIndex(1)
        self.wait(200)
        self.grab_brain("after_apply_all")
        self.ok(self.w._glm_tabs.isVisible(), "GLM tabs visible")

    def _test_close_reopen(self):
        self.S("CLOSE file")
        self.w._on_close_file()
        self.wait(300)
        self.grab("closed")
        self.ok(self.w._current_data is None, "Data cleared")
        self.ok(not self.w._glm_tabs.isVisible(), "GLM hidden")

        self.S("REOPEN manual")
        self.w._processing_panel._on_manual_clicked()
        self.wait(100)
        self.w._data_manager.load_file(SNIRF)
        self.wait(500)
        self.grab_graph("reopened_raw")
        self.ok(self.w._current_data is not None, "Data back")

        self.S("REOPEN + Apply All")
        self.w._on_apply_all()
        self.wait(500)
        self.grab_graph("reopened_conc")
        self.ok(self.w._pipeline.result.hbo is not None, "Pipeline OK")

        self.S("REOPEN + GLM")
        self.w._on_run_glm()
        self.wait(500)
        self.w._glm_tabs.setCurrentIndex(0)
        self.wait(200)
        self.grab_probe("reopened")
        self.w._glm_tabs.setCurrentIndex(1)
        self.wait(200)
        self.grab_brain("reopened")
        self.ok(self.w._glm_tabs.isVisible(), "GLM works after reopen")

    def _test_auto_mode(self):
        self.S("AUTO MODE — close + switch + load")
        self.w._on_close_file()
        self.wait(200)
        self.w._processing_panel._on_auto_clicked()
        self.wait(100)
        self.w._data_manager.load_file(SNIRF)
        self.wait(500)
        self.grab("auto_loaded")
        self.grab_graph("auto_conc")
        self.ok(getattr(self.w._graph, '_conc_mode', False), "Auto -> conc")
        self.ok(self.w._pipeline.result.hbo is not None, "Auto HbO")
        self.ok(_has_curve_data(self.w._graph), "Has data")

        self.S("AUTO — All channels")
        self.w._graph._select_all()
        self.wait(300)
        self.grab_graph("auto_all")
        self.ok(_has_curve_data(self.w._graph), "Not empty")

        self.S("AUTO — Stacked")
        self.w._graph._set_view_mode(True)
        self.wait(300)
        self.grab_graph("auto_stacked")
        self.ok(_has_curve_data(self.w._graph), "Not empty")

        self.S("AUTO — GLM")
        self.w._graph._set_view_mode(False)
        self.w._graph._spin_first.setValue(5)
        self.w._graph._select_first_n()
        self.w._on_run_glm()
        self.wait(500)
        self.grab("auto_glm")
        self.w._glm_tabs.setCurrentIndex(0)
        self.wait(200)
        self.grab_probe("auto_glm")
        self.w._glm_tabs.setCurrentIndex(1)
        self.wait(200)
        self.grab_brain("auto_glm")
        self.ok(self.w._glm_tabs.isVisible(), "Auto GLM works")


def main():
    app = QApplication(sys.argv)
    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()
    app.processEvents()
    runner = TestRunner(window, app)
    QTimer.singleShot(500, runner.run_all)
    app.exec_()
    sys.exit(1 if runner.errors else 0)


if __name__ == '__main__':
    main()
