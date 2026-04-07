"""
Tests for EarlyStopping and the evaluation helpers in engine.py.
"""
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.dl.engine import (
    EarlyStopping, create_test_results, calculate_measures,
)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_does_not_stop_on_first_call(self):
        es = EarlyStopping(patience=3)
        assert not es(0.5)

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=3)
        es(0.5)      # best = 0.5, counter = 0
        es(0.4)      # counter = 1
        es(0.4)      # counter = 2
        result = es(0.4)  # counter = 3 → stop
        assert result

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2)
        es(0.5)
        es(0.4)      # counter = 1
        assert not es(0.9)  # big improvement → counter resets to 0, does not stop

    def test_delta_prevents_early_reset(self):
        es = EarlyStopping(patience=2, delta=0.05)
        es(0.5)
        # improvement of only 0.02 — below delta → not considered improvement
        es(0.52)     # counter = 1
        assert es(0.51)  # counter = 2 → stops

    def test_best_is_updated_on_improvement(self):
        es = EarlyStopping(patience=5)
        es(0.3)
        es(0.8)
        assert es.best == 0.8

    def test_counter_increments_correctly(self):
        es = EarlyStopping(patience=5)
        es(0.5)
        es(0.3)
        es(0.2)
        assert es.counter == 2


# ---------------------------------------------------------------------------
# calculate_measures
# ---------------------------------------------------------------------------

class TestCalculateMeasures:

    def _make_df(self, true_labels, pred_labels):
        return pd.DataFrame({
            "PathoPhenotype": true_labels,
            "Predicted Label": pred_labels,
        })

    def test_perfect_predictions(self):
        df = self._make_df([1, 1, 0, 0], [1, 1, 0, 0])
        m = calculate_measures(df)
        assert m["MCC"] == pytest.approx(1.0)
        assert m["BACC"] == pytest.approx(1.0)
        assert m["Sensitivity"] == pytest.approx(1.0)
        assert m["Specificity"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        df = self._make_df([1, 0], [1, 0])
        m = calculate_measures(df)
        for key in ("MCC", "BACC", "ROCAUC", "Sensitivity", "Specificity"):
            assert key in m

    def test_all_wrong_predictions(self):
        df = self._make_df([1, 1, 0, 0], [0, 0, 1, 1])
        m = calculate_measures(df)
        assert m["MCC"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# create_test_results
# ---------------------------------------------------------------------------

class TestCreateTestResults:

    def _make_predicted_data(self):
        ep = pd.Series({
            "Binary Prediction Mean": 1,
            "Prediction Mean": 0.8,
            "Prediction STD": 0.05,
            "Phenotype": "Human Pathogenic",
        })
        return {
            "sample1": {
                "Features": {"Filename": "sample1"},
                "Ensemble Predictions": ep,
            }
        }

    def test_creates_output_folder(self, tmp_path):
        label_df = pd.DataFrame({
            "File Name": ["sample1"],
            "PathoPhenotype": [1],
        })
        out_folder = str(tmp_path / "results")
        create_test_results(label_df, self._make_predicted_data(), out_folder)
        assert os.path.isdir(out_folder)

    def test_writes_tsv(self, tmp_path):
        label_df = pd.DataFrame({
            "File Name": ["sample1"],
            "PathoPhenotype": [1],
        })
        out_folder = str(tmp_path / "results")
        create_test_results(label_df, self._make_predicted_data(), out_folder)
        assert (Path(out_folder) / "test_results.tsv").exists()

    def test_fills_predicted_label(self, tmp_path):
        label_df = pd.DataFrame({
            "File Name": ["sample1"],
            "PathoPhenotype": [1],
        })
        out_folder = str(tmp_path / "results")
        result = create_test_results(label_df, self._make_predicted_data(), out_folder)
        assert result["Predicted Label"].iloc[0] == 1
