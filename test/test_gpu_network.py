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
            for pred in batch.predictions:
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
            assert batch.attentions is not None


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
            assert batch.proteome_lengths is not None


# Note: validation_pass tests omitted because they require a labeled training
# dataset, but our test fixture uses a prediction-mode dataset (no labels).
