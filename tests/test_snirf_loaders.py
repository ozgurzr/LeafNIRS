"""Test suite — SNIRF loader comparison using real OpenNeuro ds007420 data."""
import os
import sys
import time
import unittest
import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_io.snirf_loader_lib import SNIRFLoaderLib
from data_io.snirf_loader_h5py import SNIRFLoaderH5py

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'fNIRS_1')


def _find_snirf_files():
    return sorted(glob.glob(os.path.join(DATA_ROOT, '**', '*.snirf'), recursive=True))


SNIRF_FILES = _find_snirf_files()
FIRST_SNIRF = SNIRF_FILES[0] if SNIRF_FILES else None


def _skip_if_no_data():
    if not SNIRF_FILES:
        raise unittest.SkipTest(f"No .snirf files found under {os.path.abspath(DATA_ROOT)}")


class TestSNIRFLoaderLib(unittest.TestCase):

    def setUp(self):
        _skip_if_no_data()
        self.loader = SNIRFLoaderLib()
        self.data = self.loader.load(FIRST_SNIRF)

    def test_load_no_error(self):
        self.assertIsNotNone(self.data)

    def test_intensity_is_2d(self):
        self.assertEqual(self.data.intensity.ndim, 2)

    def test_has_channels(self):
        self.assertGreater(self.data.n_channels, 0)

    def test_has_timepoints(self):
        self.assertGreater(self.data.n_timepoints, 0)

    def test_time_vector_length(self):
        self.assertEqual(len(self.data.time), self.data.n_timepoints)

    def test_probe_geometry(self):
        self.assertGreater(self.data.n_sources, 0)
        self.assertGreater(self.data.n_detectors, 0)
        self.assertGreater(len(self.data.wavelength_list), 0)

    def test_channels_have_valid_indices(self):
        for ch in self.data.channels:
            self.assertGreater(ch.source_index, 0)
            self.assertGreater(ch.detector_index, 0)
            self.assertGreater(ch.wavelength_index, 0)

    def test_sampling_rate_positive(self):
        self.assertGreater(self.data.sampling_rate, 0)

    def test_duration_positive(self):
        self.assertGreater(self.data.duration_seconds, 0)


class TestSNIRFLoaderH5py(unittest.TestCase):

    def setUp(self):
        _skip_if_no_data()
        self.loader = SNIRFLoaderH5py()
        self.data = self.loader.load(FIRST_SNIRF)

    def test_load_no_error(self):
        self.assertIsNotNone(self.data)

    def test_intensity_is_2d(self):
        self.assertEqual(self.data.intensity.ndim, 2)

    def test_has_channels(self):
        self.assertGreater(self.data.n_channels, 0)

    def test_has_timepoints(self):
        self.assertGreater(self.data.n_timepoints, 0)

    def test_probe_geometry(self):
        self.assertGreater(self.data.n_sources, 0)
        self.assertGreater(self.data.n_detectors, 0)

    def test_metadata_not_empty(self):
        self.assertGreater(len(self.data.metadata), 0)
        self.assertIn('SubjectID', self.data.metadata)


class TestLoaderComparison(unittest.TestCase):
    """Compare outputs of both loaders for consistency."""

    def setUp(self):
        _skip_if_no_data()
        self.data_a = SNIRFLoaderLib().load(FIRST_SNIRF)
        self.data_b = SNIRFLoaderH5py().load(FIRST_SNIRF)

    def test_intensity_match(self):
        self.assertTrue(np.allclose(self.data_a.intensity, self.data_b.intensity))

    def test_time_match(self):
        self.assertTrue(np.allclose(self.data_a.time, self.data_b.time))

    def test_source_positions_match(self):
        self.assertTrue(np.allclose(self.data_a.probe.source_pos, self.data_b.probe.source_pos))

    def test_detector_positions_match(self):
        self.assertTrue(np.allclose(self.data_a.probe.detector_pos, self.data_b.probe.detector_pos))

    def test_wavelengths_match(self):
        self.assertTrue(np.allclose(self.data_a.probe.wavelengths, self.data_b.probe.wavelengths))

    def test_channel_count_match(self):
        self.assertEqual(self.data_a.n_channels, self.data_b.n_channels)

    def test_channel_indices_match(self):
        for i, (a, b) in enumerate(zip(self.data_a.channels, self.data_b.channels)):
            self.assertEqual(a.source_index, b.source_index)
            self.assertEqual(a.detector_index, b.detector_index)
            self.assertEqual(a.wavelength_index, b.wavelength_index)

    def test_stimulus_count_match(self):
        self.assertEqual(len(self.data_a.stimuli), len(self.data_b.stimuli))


class TestPerformance(unittest.TestCase):

    def setUp(self):
        _skip_if_no_data()

    def test_load_time_comparison(self):
        n_runs = 5
        loader_a, loader_b = SNIRFLoaderLib(), SNIRFLoaderH5py()
        loader_a.load(FIRST_SNIRF)
        loader_b.load(FIRST_SNIRF)

        times_a, times_b = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            loader_a.load(FIRST_SNIRF)
            times_a.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            loader_b.load(FIRST_SNIRF)
            times_b.append(time.perf_counter() - t0)

        mean_a = np.mean(times_a) * 1000
        mean_b = np.mean(times_b) * 1000
        ratio = mean_a / mean_b if mean_b > 0 else float('inf')
        winner = "B (h5py)" if ratio > 1 else "A (snirf lib)"
        print(f"\n  Method A: {mean_a:.1f}ms | Method B: {mean_b:.1f}ms | Winner: {winner} ({max(ratio, 1/ratio):.1f}x)")


class TestMultipleFiles(unittest.TestCase):

    def setUp(self):
        _skip_if_no_data()

    def test_all_files_method_a(self):
        loader = SNIRFLoaderLib()
        for fp in SNIRF_FILES[:5]:
            with self.subTest(file=os.path.basename(fp)):
                data = loader.load(fp)
                self.assertGreater(data.n_channels, 0)

    def test_all_files_method_b(self):
        loader = SNIRFLoaderH5py()
        for fp in SNIRF_FILES[:5]:
            with self.subTest(file=os.path.basename(fp)):
                data = loader.load(fp)
                self.assertGreater(data.n_channels, 0)


class TestErrorHandling(unittest.TestCase):

    def test_file_not_found_lib(self):
        with self.assertRaises(FileNotFoundError):
            SNIRFLoaderLib().load("nonexistent.snirf")

    def test_file_not_found_h5py(self):
        with self.assertRaises(FileNotFoundError):
            SNIRFLoaderH5py().load("nonexistent.snirf")

    def test_invalid_extension_lib(self):
        with self.assertRaises(ValueError):
            SNIRFLoaderLib().load(__file__)

    def test_invalid_extension_h5py(self):
        with self.assertRaises(ValueError):
            SNIRFLoaderH5py().load(__file__)


if __name__ == "__main__":
    print(f"Found {len(SNIRF_FILES)} .snirf files")
    unittest.main(verbosity=2)
