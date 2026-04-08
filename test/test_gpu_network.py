"""
GPU network tests — real forward and backward passes through the neural network.

Requires: GPU (or CPU), real model weights.
Mark: @pytest.mark.slow
"""
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
TEST_GENOME = REPO_ROOT / "test" / "data" / "GCF_000014385.1_ASM1438v1_genomic.fna"


def _get_real_config():
    from pathogenfinder2.utils.configuration import ConfigurationPF2
    return ConfigurationPF2(mode="Prediction")


def _create_real_network(tmp_path):
    from pathogenfinder2.dl.utils.network import NetworkModule
    from pathogenfinder2.dl.models.convnext_addatt import ConvNext_AddAtt_Net
    cfg = _get_real_config()
    mp = cfg["Model Parameters"]
    nm = NetworkModule(
        model_type=ConvNext_AddAtt_Net,
        model_parameters=mp,
        out_folder=str(tmp_path),
        mixed_precision=mp["Mixed Precision"],
        loss_type=mp["Loss Function"],
    )
    nm.load_model(str(mp["Network Weights"][0]))
    return nm, cfg


def _create_real_dataloader(tmp_path):
    from pathogenfinder2.main import PathogenFinder2
    from pathogenfinder2.dl.utils.data import EmbeddingData
    pf2 = PathogenFinder2(mode="Prediction", outPath=str(tmp_path / "prep"))
    pf2.files_module.create_nestedoutput(
        format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
    pf2.predict_protein_content()
    pf2.infer_embeddings()
    input_df = pf2.files_module.save_inputmetadata()
    dataset = EmbeddingData.create_dataset(
        input_type="protein_embeddings",
        data_df=input_df,
        data_type="prediction",
        dual_pred=False,
    )
    mp = pf2.pf2_config["Model Parameters"]
    loader = EmbeddingData.load_data(
        dataset, batch_size=mp["Batch Size"],
        num_workers=mp["Data Parameters"]["num_workers"],
        asynchronity=mp["Data Parameters"]["asynchronity"],
    )
    return loader, pf2


@pytest.mark.slow
class TestPredictivePass:

    def test_predictions_between_0_and_1(self, tmp_path):
        loader, pf2 = _create_real_dataloader(tmp_path)
        nm, cfg = _create_real_network(tmp_path / "net")
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

    def test_attentions_returned(self, tmp_path):
        loader, pf2 = _create_real_dataloader(tmp_path)
        nm, cfg = _create_real_network(tmp_path / "net")
        mp = cfg["Model Parameters"]
        results = nm.predictive_pass(
            loader, batch_size=mp["Batch Size"],
            record_attentions=True, record_embeddings=False,
        )
        for batch in results:
            assert batch.attention is not None


@pytest.mark.slow
class TestPredictivePassEmbeddings:

    def test_embeddings_returned(self, tmp_path):
        loader, pf2 = _create_real_dataloader(tmp_path)
        nm, cfg = _create_real_network(tmp_path / "net")
        mp = cfg["Model Parameters"]
        results = nm.predictive_pass(
            loader, batch_size=mp["Batch Size"],
            record_attentions=False, record_embeddings=True,
        )
        for batch in results:
            assert batch.proteome_length is not None


@pytest.mark.slow
class TestTrainPass:

    def test_single_epoch_loss_is_finite(self, tmp_path):
        loader, pf2 = _create_real_dataloader(tmp_path)
        nm, cfg = _create_real_network(tmp_path / "net")
        optimizer = torch.optim.Adam(nm.network.parameters(), lr=1e-4)
        nm.network.train()
        loss_val, mcc_val = nm.train_pass(loader, optimizer, accumulate_gradient=1)
        assert np.isfinite(loss_val), f"Training loss is not finite: {loss_val}"


@pytest.mark.slow
class TestValidationPass:

    def test_validation_loss_is_finite(self, tmp_path):
        loader, pf2 = _create_real_dataloader(tmp_path)
        nm, cfg = _create_real_network(tmp_path / "net")
        loss_val, mcc_val = nm.validation_pass(loader)
        assert np.isfinite(loss_val), f"Validation loss is not finite: {loss_val}"

    def test_no_gradient_updates(self, tmp_path):
        loader, pf2 = _create_real_dataloader(tmp_path)
        nm, cfg = _create_real_network(tmp_path / "net")
        param = next(nm.network.parameters())
        before = param.clone().detach()
        nm.validation_pass(loader)
        after = param.clone().detach()
        assert torch.equal(before, after), "Weights changed during validation pass"
