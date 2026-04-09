"""
GPU network tests — real forward and backward passes.
Uses session-scoped fixtures from conftest.py.
"""
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.slow
class TestPredictivePass:

    def test_predictions_between_0_and_1(self, gpu_network):
        nm, loader, cfg = gpu_network
        mp = cfg["Model Parameters"]
        results = nm.predictive_pass(
            loader, batch_size=mp["Batch Size"],
            record_attentions=False, record_embeddings=False,
        )
        assert len(results) > 0
        for batch in results:
            for pred in batch.prediction:
                assert 0.0 <= pred <= 1.0


@pytest.mark.slow
class TestPredictivePassAttentions:

    def test_attentions_returned(self, gpu_network):
        nm, loader, cfg = gpu_network
        mp = cfg["Model Parameters"]
        results = nm.predictive_pass(
            loader, batch_size=mp["Batch Size"],
            record_attentions=True, record_embeddings=False,
        )
        assert len(results) > 0
        for batch in results:
            assert batch.attention is not None


@pytest.mark.slow
class TestPredictivePassEmbeddings:

    def test_embeddings_returned(self, gpu_network):
        nm, loader, cfg = gpu_network
        mp = cfg["Model Parameters"]
        results = nm.predictive_pass(
            loader, batch_size=mp["Batch Size"],
            record_attentions=False, record_embeddings=True,
        )
        assert len(results) > 0
        for batch in results:
            assert batch.proteome_length is not None


@pytest.mark.slow
class TestValidationPass:

    def test_validation_loss_is_finite(self, gpu_network):
        nm, loader, cfg = gpu_network
        loss_val, mcc_val = nm.validation_pass(loader, batch_size=cfg["Model Parameters"]["Batch Size"])
        assert np.isfinite(loss_val), f"Validation loss is not finite: {loss_val}"

    def test_no_gradient_updates(self, gpu_network):
        nm, loader, cfg = gpu_network
        param = next(nm.network.parameters())
        before = param.clone().detach()
        nm.validation_pass(loader, batch_size=cfg["Model Parameters"]["Batch Size"])
        after = param.clone().detach()
        assert torch.equal(before, after), "Weights changed during validation pass"
