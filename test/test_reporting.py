"""
Tests for BatchResults and FileReport.
"""
import pickle
import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.dl.utils.reporting import BatchResults, FileReport


# ---------------------------------------------------------------------------
# BatchResults
# ---------------------------------------------------------------------------

def _make_batch(batch_size=2, n_models=1):
    filenames = [f"sample{i}.h5" for i in range(batch_size)]
    predictions = torch.tensor([[0.7], [0.3]] if batch_size == 2 else [[0.7]])
    prot_ids = [np.array(["prot1", "prot2"]) for _ in range(batch_size)]
    proteome_lengths = torch.tensor([[10], [8]] if batch_size == 2 else [[10]])
    return BatchResults(
        filenames=filenames,
        predictions=predictions,
        protIDs=prot_ids,
        proteome_lengths=proteome_lengths,
    )


class TestBatchResults:

    def test_len(self):
        br = _make_batch(2)
        assert len(br) == 2

    def test_get_samples_keys(self):
        br = _make_batch(1)
        samples = br.get_samples()
        assert "sample0.h5" in samples
        sample = samples["sample0.h5"]
        assert "Features" in sample
        assert "Output" in sample

    def test_prediction_list(self):
        br = _make_batch(1)
        samples = br.get_samples()
        preds = samples["sample0.h5"]["Output"]["Prediction"]
        assert isinstance(preds, list)
        assert abs(preds[0] - 0.7) < 1e-5

    def test_proteome_length_stored(self):
        br = _make_batch(1)
        samples = br.get_samples()
        assert samples["sample0.h5"]["Features"]["Proteome Length"] == 10

    def test_attention_none_when_not_provided(self):
        br = _make_batch(1)
        samples = br.get_samples()
        assert samples["sample0.h5"]["Output"]["Attention"] == [None]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(AssertionError):
            BatchResults(
                filenames=["a", "b"],
                predictions=torch.tensor([[0.5]]),  # wrong length
                protIDs=[np.array(["p1"]), np.array(["p2"])],
                proteome_lengths=torch.tensor([[5], [6]]),
            )

    def test_attention_stored_when_provided(self):
        br = BatchResults(
            filenames=["s.h5"],
            predictions=torch.tensor([[0.6]]),
            protIDs=[np.array(["p1", "p2"])],
            proteome_lengths=torch.tensor([[2]]),
            attentions=torch.zeros(1, 4, 10),
        )
        samples = br.get_samples()
        att = samples["s.h5"]["Output"]["Attention"]
        assert isinstance(att, np.ndarray)


# ---------------------------------------------------------------------------
# FileReport
# ---------------------------------------------------------------------------

class TestFileReport:

    def test_finish_report_writes_pickle(self, tmp_path):
        report = FileReport(
            configuration={"Model Name": "ConvNext-AddAtt"},
            name="run1",
            dict_dir=str(tmp_path),
            modes="Train",
        )
        report.add_epoch_info({"Epoch": 0, "Training Loss": 0.5,
                               "Validation Loss": 0.4, "Training MCC": 0.1,
                               "Validation MCC": 0.2,
                               "Epoch Runtime (seconds)": 1.0})
        report.finish_report()
        pkl_path = tmp_path / "run1_results.pickle"
        assert pkl_path.exists()
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        assert "Train" in data
        assert "Epoch 0" in data["Train"]

    def test_configuration_stored(self, tmp_path):
        cfg = {"Model Name": "FNN", "Batch Size": 16}
        report = FileReport(configuration=cfg, name="r", dict_dir=str(tmp_path), modes="Prediction")
        assert report.data["Configuration"] == cfg

    def test_start_train_report_noop(self, tmp_path):
        """start_train_report must not raise (it's a no-op for FileReport)."""
        report = FileReport(configuration={}, name="r", dict_dir=str(tmp_path), modes="Train")
        report.start_train_report(model=None, criterion=None)

    def test_add_step_info_noop(self, tmp_path):
        """add_step_info must not raise (it's a no-op for FileReport)."""
        report = FileReport(configuration={}, name="r", dict_dir=str(tmp_path), modes="Train")
        report.add_step_info(loss_train=0.3, lr=1e-4, batch_n=0, len_dataloader=10)
