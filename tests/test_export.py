"""Tests for export_manager and group_glm modules."""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from processing.export_manager import export_csv, export_matlab
from processing.glm_analysis import GLMResult
from processing.group_glm import run_group_glm, GroupGLMResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dummy_data(n_time=200, n_pairs=4, n_conditions=2):
    """Generate synthetic pipeline data for testing."""
    rng = np.random.RandomState(42)
    time = np.linspace(0, 100, n_time)
    hbo = rng.randn(n_time, n_pairs) * 0.01
    hbr = rng.randn(n_time, n_pairs) * 0.005
    pair_labels = [f"S{i+1}-D{i+1}" for i in range(n_pairs)]

    glm = GLMResult(
        beta=rng.randn(n_conditions, n_pairs),
        t_stat=rng.randn(n_conditions, n_pairs) * 3,
        p_value=rng.rand(n_conditions, n_pairs),
        contrast_names=[f"cond_{i+1}" for i in range(n_conditions)],
        pair_labels=pair_labels,
        design_matrix=rng.randn(n_time, n_conditions + 2),
        residuals=rng.randn(n_time, n_pairs),
        df=n_time - n_conditions - 2,
    )
    return time, hbo, hbr, pair_labels, glm


# ---------------------------------------------------------------------------
# CSV Export Tests
# ---------------------------------------------------------------------------

class TestCSVExport:

    def test_csv_timeseries_created(self, tmp_path):
        time, hbo, hbr, labels, glm = _make_dummy_data()
        files = export_csv(
            filepath=str(tmp_path / "test.csv"),
            time=time, data=hbo, data_label="HbO / HbR",
            pair_labels=labels, hbr=hbr,
        )
        assert len(files) == 1
        assert os.path.isfile(files[0])

    def test_csv_timeseries_row_count(self, tmp_path):
        time, hbo, hbr, labels, _ = _make_dummy_data(n_time=150)
        files = export_csv(
            filepath=str(tmp_path / "test.csv"),
            time=time, data=hbo, data_label="HbO / HbR",
            pair_labels=labels, hbr=hbr,
        )
        import pandas as pd
        df = pd.read_csv(files[0])
        assert len(df) == 150

    def test_csv_columns_concentration(self, tmp_path):
        time, hbo, hbr, labels, _ = _make_dummy_data(n_pairs=3)
        files = export_csv(
            filepath=str(tmp_path / "test.csv"),
            time=time, data=hbo, data_label="HbO / HbR",
            pair_labels=labels, hbr=hbr,
        )
        import pandas as pd
        df = pd.read_csv(files[0])
        assert "time_s" in df.columns
        assert "HbO_S1-D1" in df.columns
        assert "HbR_S1-D1" in df.columns
        # 1 time + 3 HbO + 3 HbR = 7
        assert len(df.columns) == 7

    def test_csv_with_glm(self, tmp_path):
        time, hbo, hbr, labels, glm = _make_dummy_data()
        files = export_csv(
            filepath=str(tmp_path / "test.csv"),
            time=time, data=hbo, data_label="HbO / HbR",
            pair_labels=labels, hbr=hbr,
            glm_hbo=glm, glm_hbr=glm,
        )
        assert len(files) == 2
        assert any("glm" in f for f in files)

    def test_csv_raw_mode(self, tmp_path):
        """Export raw intensity (non-concentration) mode."""
        rng = np.random.RandomState(0)
        time = np.linspace(0, 10, 100)
        data = rng.randn(100, 8)
        files = export_csv(
            filepath=str(tmp_path / "raw.csv"),
            time=time, data=data, data_label="Raw Intensity",
        )
        import pandas as pd
        df = pd.read_csv(files[0])
        assert len(df) == 100
        assert len(df.columns) == 9  # 1 time + 8 channels


# ---------------------------------------------------------------------------
# MATLAB Export Tests
# ---------------------------------------------------------------------------

class TestMATLABExport:

    def test_mat_file_created(self, tmp_path):
        time, hbo, hbr, labels, glm = _make_dummy_data()
        path = export_matlab(
            filepath=str(tmp_path / "test.mat"),
            time=time, data=hbo, data_label="HbO / HbR",
            sampling_rate=10.0, pair_labels=labels, hbr=hbr,
        )
        assert os.path.isfile(path)

    def test_mat_roundtrip(self, tmp_path):
        time, hbo, hbr, labels, glm = _make_dummy_data()
        path = export_matlab(
            filepath=str(tmp_path / "test.mat"),
            time=time, data=hbo, data_label="HbO / HbR",
            sampling_rate=10.0, pair_labels=labels, hbr=hbr,
            glm_hbo=glm,
        )
        from scipy.io import loadmat
        mat = loadmat(path)
        assert "time" in mat
        assert "hbo" in mat
        assert "hbr" in mat
        np.testing.assert_array_almost_equal(mat["time"].ravel(), time)
        np.testing.assert_array_almost_equal(mat["hbo"], hbo)

    def test_mat_glm_variables(self, tmp_path):
        time, hbo, hbr, labels, glm = _make_dummy_data()
        path = export_matlab(
            filepath=str(tmp_path / "test.mat"),
            time=time, data=hbo, data_label="HbO / HbR",
            sampling_rate=10.0, pair_labels=labels,
            glm_hbo=glm, glm_hbr=glm,
        )
        from scipy.io import loadmat
        mat = loadmat(path)
        assert "glm_hbo_beta" in mat
        assert "glm_hbo_tstat" in mat
        assert "glm_hbo_pvalue" in mat
        assert "glm_hbr_beta" in mat
        assert mat["glm_hbo_beta"].shape == glm.beta.shape


# ---------------------------------------------------------------------------
# Group GLM Tests
# ---------------------------------------------------------------------------

class TestGroupGLM:

    def test_requires_two_subjects(self):
        _, _, _, _, glm = _make_dummy_data()
        with pytest.raises(ValueError, match="at least 2"):
            run_group_glm([glm])

    def test_basic_group_glm(self):
        results = []
        for seed in range(5):
            rng = np.random.RandomState(seed)
            glm = GLMResult(
                beta=rng.randn(2, 4) + 2.0,  # strong positive effect
                t_stat=np.zeros((2, 4)),
                p_value=np.zeros((2, 4)),
                contrast_names=["a", "b"],
                pair_labels=["S1-D1", "S1-D2", "S2-D1", "S2-D2"],
                design_matrix=np.zeros((10, 4)),
                residuals=np.zeros((10, 4)),
                df=8,
            )
            results.append(glm)

        group = run_group_glm(results)
        assert isinstance(group, GroupGLMResult)
        assert group.n_subjects == 5
        assert group.group_t_stat.shape == (2, 4)
        assert group.group_p_value.shape == (2, 4)
        assert group.subject_betas.shape == (5, 2, 4)

    def test_significant_effect_detected(self):
        """A large consistent effect should be detected as significant."""
        results = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            glm = GLMResult(
                beta=np.full((1, 2), 5.0) + rng.randn(1, 2) * 0.5,
                t_stat=np.zeros((1, 2)),
                p_value=np.zeros((1, 2)),
                contrast_names=["task"],
                pair_labels=["S1-D1", "S2-D2"],
                design_matrix=np.zeros((10, 3)),
                residuals=np.zeros((10, 2)),
                df=7,
            )
            results.append(glm)

        group = run_group_glm(results)
        assert np.all(group.group_p_value < 0.05), \
            f"Expected all significant, got p={group.group_p_value}"

    def test_null_effect_not_significant(self):
        """Zero-mean betas should not be significant with few subjects."""
        results = []
        for seed in range(4):
            rng = np.random.RandomState(seed)
            glm = GLMResult(
                beta=rng.randn(1, 2) * 0.01,  # near-zero effect
                t_stat=np.zeros((1, 2)),
                p_value=np.zeros((1, 2)),
                contrast_names=["task"],
                pair_labels=["S1-D1", "S2-D2"],
                design_matrix=np.zeros((10, 3)),
                residuals=np.zeros((10, 2)),
                df=7,
            )
            results.append(glm)

        group = run_group_glm(results)
        # With near-zero effect and 4 subjects, should NOT be significant
        assert np.all(group.group_p_value > 0.01), \
            f"Expected non-significant, got p={group.group_p_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
