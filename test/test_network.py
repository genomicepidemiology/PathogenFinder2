"""
Tests for NetworkModule static/class methods and load/save helpers.
"""
import os
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.dl.utils.network import NetworkModule


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:

    def test_returns_string(self):
        device = NetworkModule.get_device()
        assert isinstance(device, str)

    def test_valid_device_value(self):
        device = NetworkModule.get_device()
        assert device in ("cuda", "cpu")

    @patch("pathogenfinder2.dl.utils.network.torch.cuda.is_available", return_value=True)
    def test_returns_cuda_when_available(self, _mock):
        assert NetworkModule.get_device() == "cuda"

    @patch("pathogenfinder2.dl.utils.network.torch.cuda.is_available", return_value=False)
    def test_returns_cpu_when_no_cuda(self, _mock):
        assert NetworkModule.get_device() == "cpu"


# ---------------------------------------------------------------------------
# set_sensible_batch_size
# ---------------------------------------------------------------------------

class TestSetSensibleBatchSize:

    def test_exact_multiple_of_8(self):
        batch, accum = NetworkModule.set_sensible_batch_size(64)
        assert batch * accum == 64
        assert batch <= 45

    def test_large_power_of_two(self):
        batch, accum = NetworkModule.set_sensible_batch_size(128)
        assert batch * accum == 128

    def test_returns_integers(self):
        batch, accum = NetworkModule.set_sensible_batch_size(32)
        assert float(accum).is_integer()

    def test_non_multiple_raises(self):
        with pytest.raises(ValueError, match="not a multiple"):
            NetworkModule.set_sensible_batch_size(50)

    def test_below_min_raises(self):
        with pytest.raises(ValueError, match="smaller than the minimum"):
            NetworkModule.set_sensible_batch_size(4)

    def test_batch_within_bounds(self):
        batch, _ = NetworkModule.set_sensible_batch_size(64)
        assert 8 <= batch <= 45


# ---------------------------------------------------------------------------
# NetworkModule instantiation (CPU, minimal config)
# ---------------------------------------------------------------------------

def _minimal_params():
    return {
        "Input Dimensions": 1024,
        "Out Dimensions": 1,
        "Seed": None,
        "Norm Scale": 1e-6,
        "Norm Type": "Layer",
        "Attention Norm": True,
        "Stochastic Depth Prob": 0.0,
        "Stochastic Depth Prob Att": False,
        "Sequence Dropout": 0.0,
        "Attention Dropout": 0.0,
        "Network Structure": {
            "Stem Cell": True,
            "Num Blocks": 1,
            "Block Dimensions": 16,
            "Attention Dimensions": 8,
            "FNN Dimensions": 0,
            "Length Information": "concat1",
            "Length Dimensions": 4,
            "Residual Attention": False,
        },
    }


def _make_minimal_network(tmp_path=None):
    """Build a NetworkModule with a tiny ConvNext-AddAtt on CPU."""
    from pathogenfinder2.dl.models.convnext_addatt import ConvNext_AddAtt_Net
    nm = NetworkModule(
        model_type=ConvNext_AddAtt_Net,
        model_parameters=_minimal_params(),
        out_folder=str(tmp_path) if tmp_path else "/tmp",
        device="cpu",
        mixed_precision=False,
        results_module=None,
        memory_profiler=None,
        loss_type="bcelogits",
    )
    return nm


class TestNetworkModuleInit:

    def test_device_is_cpu(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        assert nm.device == "cpu"

    def test_network_is_not_none(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        assert nm.network is not None

    def test_loss_function_bcelogits(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        assert isinstance(nm.loss_function, torch.nn.BCEWithLogitsLoss)

    def test_loss_function_bce(self, tmp_path):
        from pathogenfinder2.dl.models.convnext_addatt import ConvNext_AddAtt_Net
        params = _minimal_params()
        nm = NetworkModule(
            model_type=ConvNext_AddAtt_Net, model_parameters=params,
            out_folder=str(tmp_path), device="cpu", mixed_precision=False,
            results_module=None, memory_profiler=None, loss_type="bce",
        )
        assert isinstance(nm.loss_function, torch.nn.BCELoss)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestSaveLoadModel:

    def test_save_creates_file(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        nm.save_model(loss=0.5, mcc_val=0.8, epoch=1)
        assert (tmp_path / "weights_model.pt").exists()

    def test_load_weights_returns_none(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        nm.save_model()
        result = nm.load_model(str(tmp_path / "weights_model.pt"))
        assert result is None

    def test_load_checkpoint_returns_dict(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        optimizer = torch.optim.Adam(nm.network.parameters(), lr=1e-4)
        # Manually save a checkpoint with optimizer state
        torch.save({
            "model_state_dict": nm.network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 5,
            "loss": 0.3,
        }, str(tmp_path / "ckpt.pt"))
        result = nm.load_model(str(tmp_path / "ckpt.pt"))
        assert result is not None
        assert result["Epoch"] == 5

    def test_roundtrip_weights(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        # Record initial state
        before = {k: v.clone() for k, v in nm.network.state_dict().items()}
        nm.save_model()
        # Randomise weights
        for p in nm.network.parameters():
            torch.nn.init.uniform_(p)
        # Reload
        nm.load_model(str(tmp_path / "weights_model.pt"))
        after = nm.network.state_dict()
        for k in before:
            assert torch.allclose(before[k].float(), after[k].float())


# ---------------------------------------------------------------------------
# calculate_loss
# ---------------------------------------------------------------------------

class TestCalculateLoss:

    def test_loss_is_scalar(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        preds = torch.randn(4, 1)
        labels = torch.randint(0, 2, (4, 1)).float()
        _, loss = nm.calculate_loss(preds, labels)
        assert loss.shape == torch.Size([])

    def test_predictions_in_0_1(self, tmp_path):
        nm = _make_minimal_network(tmp_path)
        preds = torch.randn(4, 1)
        labels = torch.randint(0, 2, (4, 1)).float()
        predictions, _ = nm.calculate_loss(preds, labels)
        assert (predictions >= 0).all() and (predictions <= 1).all()
