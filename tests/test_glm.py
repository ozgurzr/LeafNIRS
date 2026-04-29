"""Unit tests for GLM statistical analysis."""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from processing.glm_analysis import canonical_hrf, build_design_matrix, run_glm, compute_glm


class FakeStimulus:
    def __init__(self, name, onset, amplitude=None):
        self.name = name
        self.onset = np.array(onset)
        self.amplitude = np.array(amplitude) if amplitude is not None else np.ones(len(onset))


class TestCanonicalHRF(unittest.TestCase):
    def test_hrf_shape(self):
        t = np.arange(0, 32, 0.1)
        h = canonical_hrf(t)
        self.assertEqual(h.shape, t.shape)

    def test_peak_at_5s(self):
        t = np.arange(0, 32, 0.01)
        h = canonical_hrf(t)
        peak_t = t[np.argmax(h)]
        self.assertAlmostEqual(peak_t, 5.0, delta=1.5)

    def test_undershoot_after_peak(self):
        t = np.arange(0, 32, 0.1)
        h = canonical_hrf(t)
        idx_10_20 = (t >= 10) & (t <= 25)
        self.assertTrue(np.any(h[idx_10_20] < 0), "HRF should have negative undershoot")

    def test_unit_peak(self):
        t = np.arange(0, 32, 0.1)
        h = canonical_hrf(t)
        self.assertAlmostEqual(np.max(h), 1.0, places=5)

    def test_zero_at_start(self):
        h = canonical_hrf(np.array([0.0]))
        self.assertAlmostEqual(h[0], 0.0, places=3)


class TestDesignMatrix(unittest.TestCase):
    def setUp(self):
        self.fs = 10.0
        self.duration = 120.0
        self.time = np.arange(0, self.duration, 1.0 / self.fs)
        self.stimuli = [
            FakeStimulus("TaskA", [10, 40, 70]),
            FakeStimulus("TaskB", [25, 55, 85]),
        ]

    def test_shape(self):
        X, names = build_design_matrix(self.time, self.stimuli, self.fs, add_drift=1)
        # 2 conditions + constant + linear drift = 4 columns
        self.assertEqual(X.shape[0], len(self.time))
        self.assertEqual(X.shape[1], 4)

    def test_regressor_names(self):
        X, names = build_design_matrix(self.time, self.stimuli, self.fs, add_drift=1)
        self.assertEqual(names, ["TaskA", "TaskB", "constant", "drift_order1"])

    def test_convolved_shape(self):
        X, names = build_design_matrix(self.time, self.stimuli, self.fs)
        # Each column should be same length as time
        for col in range(X.shape[1]):
            self.assertEqual(X.shape[0], len(self.time))

    def test_stimulus_columns_not_zero(self):
        X, names = build_design_matrix(self.time, self.stimuli, self.fs)
        # Stimulus columns should have non-zero values
        self.assertTrue(np.any(X[:, 0] != 0), "TaskA column is all zeros")
        self.assertTrue(np.any(X[:, 1] != 0), "TaskB column is all zeros")

    def test_no_drift(self):
        X, names = build_design_matrix(self.time, self.stimuli, self.fs, add_drift=0)
        self.assertEqual(X.shape[1], 3)  # 2 conditions + constant

    def test_quadratic_drift(self):
        X, names = build_design_matrix(self.time, self.stimuli, self.fs, add_drift=2)
        self.assertEqual(X.shape[1], 5)  # 2 cond + const + linear + quadratic


class TestGLMSolver(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_time = 1000
        self.n_pairs = 10
        self.fs = 10.0

    def test_known_betas(self):
        """Test GLM recovery of known beta weights."""
        n = self.n_time
        X = np.column_stack([
            np.sin(2 * np.pi * 0.1 * np.arange(n) / self.fs),
            np.cos(2 * np.pi * 0.2 * np.arange(n) / self.fs),
            np.ones(n),
        ])
        true_beta = np.array([[2.0, 1.5, 0.8, 0.3, 2.5,
                                1.0, 0.5, 1.2, 0.7, 1.8]])
        # Y = X @ beta + noise
        Y_clean = X[:, [0]] * true_beta
        noise = np.random.randn(n, self.n_pairs) * 0.01
        Y = Y_clean + noise

        result = run_glm(Y, X, [0], ["cond1"],
                         [f"P{i}" for i in range(self.n_pairs)])

        # Recovered betas should be close to true
        for i in range(self.n_pairs):
            self.assertAlmostEqual(result.beta[0, i], true_beta[0, i], delta=0.1)

    def test_t_statistics_significant(self):
        """Strong signal should produce significant t-stats."""
        n = self.n_time
        signal = np.sin(2 * np.pi * 0.05 * np.arange(n) / self.fs)
        X = np.column_stack([signal, np.ones(n)])
        Y = signal[:, None] * 5.0 + np.random.randn(n, 1) * 0.1

        result = run_glm(Y, X, [0], ["sig"],
                         ["P1"])
        self.assertGreater(abs(result.t_stat[0, 0]), 10)
        self.assertLess(result.p_value[0, 0], 0.001)

    def test_null_signal_not_significant(self):
        """Pure noise should not be significant."""
        n = self.n_time
        X = np.column_stack([np.random.randn(n), np.ones(n)])
        Y = np.random.randn(n, 3) * 1.0

        result = run_glm(Y, X, [0], ["noise"],
                         ["P1", "P2", "P3"])
        # Most p-values should be > 0.05
        self.assertGreater(np.mean(result.p_value > 0.05), 0.5)

    def test_degrees_of_freedom(self):
        n = self.n_time
        n_regressors = 4
        X = np.random.randn(n, n_regressors)
        Y = np.random.randn(n, 2)
        result = run_glm(Y, X, [0, 1], ["c1", "c2"], ["P1", "P2"])
        self.assertEqual(result.df, n - n_regressors)

    def test_output_shapes(self):
        n = self.n_time
        X = np.column_stack([np.random.randn(n), np.random.randn(n), np.ones(n)])
        Y = np.random.randn(n, 5)
        result = run_glm(Y, X, [0, 1], ["c1", "c2"],
                         [f"P{i}" for i in range(5)])
        self.assertEqual(result.beta.shape, (2, 5))
        self.assertEqual(result.t_stat.shape, (2, 5))
        self.assertEqual(result.p_value.shape, (2, 5))
        self.assertEqual(result.residuals.shape, (n, 5))


class TestComputeGLM(unittest.TestCase):
    def test_full_pipeline(self):
        """Test the high-level compute_glm function."""
        np.random.seed(42)
        fs = 10.0
        duration = 60.0
        n_time = int(duration * fs)
        n_pairs = 5
        time = np.arange(n_time) / fs

        stimuli = [FakeStimulus("cond1", [5, 25, 45])]
        hbo = np.random.randn(n_time, n_pairs) * 0.1
        hbr = np.random.randn(n_time, n_pairs) * 0.05
        pair_labels = [f"S1-D{i+1}" for i in range(n_pairs)]

        glm_hbo, glm_hbr = compute_glm(hbo, hbr, time, stimuli, pair_labels, fs)

        self.assertEqual(glm_hbo.beta.shape[0], 1)  # 1 condition
        self.assertEqual(glm_hbo.beta.shape[1], n_pairs)
        self.assertEqual(glm_hbr.beta.shape, glm_hbo.beta.shape)
        self.assertEqual(glm_hbo.contrast_names, ["cond1"])
        self.assertEqual(glm_hbo.pair_labels, pair_labels)


if __name__ == '__main__':
    unittest.main()
