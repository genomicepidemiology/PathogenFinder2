"""
GPU inference tests — PathogenDLModel.predict_model() with real weights.

Requires: GPU (or CPU with patience), real model weights shipped with package.
Mark: @pytest.mark.slow
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
TEST_GENOME = REPO_ROOT / "test" / "data" / "GCF_000014385.1_ASM1438v1_genomic.fna"


def _create_embedding_file(tmp_path):
    from pathogenfinder2.main import PathogenFinder2
    pf2 = PathogenFinder2(mode="Prediction", outPath=str(tmp_path / "prep"))
    pf2.files_module.create_nestedoutput(
        format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
    pf2.predict_protein_content()
    pf2.infer_embeddings()
    input_df = pf2.files_module.save_inputmetadata()
    return pf2, input_df


@pytest.mark.slow
class TestPredictModel:

    def test_returns_dict_with_predictions(self, tmp_path):
        from pathogenfinder2.dl.model import PathogenDLModel
        pf2, input_df = _create_embedding_file(tmp_path)
        model_dl = PathogenDLModel(
            model_parameters=pf2.pf2_config["Model Parameters"],
            misc_parameters=pf2.pf2_config["Misc Parameters"],
            seed=pf2.pf2_config["Model Parameters"]["Seed"],
        )
        results = model_dl.predict_model(input_metapath=input_df)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_predictions_are_valid_floats(self, tmp_path):
        from pathogenfinder2.dl.model import PathogenDLModel
        pf2, input_df = _create_embedding_file(tmp_path)
        model_dl = PathogenDLModel(
            model_parameters=pf2.pf2_config["Model Parameters"],
            misc_parameters=pf2.pf2_config["Misc Parameters"],
            seed=pf2.pf2_config["Model Parameters"]["Seed"],
        )
        results = model_dl.predict_model(input_metapath=input_df)
        for sample_name, sample_data in results.items():
            preds = sample_data["Output"]["Prediction"]
            assert len(preds) == 4, f"Expected 4 ensemble predictions, got {len(preds)}"
            for p in preds:
                assert 0.0 <= p <= 1.0, f"Prediction {p} out of range [0, 1]"


@pytest.mark.slow
class TestPredictModelEmbeddings:

    def test_embeddings_captured(self, tmp_path):
        from pathogenfinder2.dl.model import PathogenDLModel
        pf2, input_df = _create_embedding_file(tmp_path)
        model_dl = PathogenDLModel(
            model_parameters=pf2.pf2_config["Model Parameters"],
            misc_parameters=pf2.pf2_config["Misc Parameters"],
            seed=pf2.pf2_config["Model Parameters"]["Seed"],
        )
        results = model_dl.predict_model(input_metapath=input_df, save_embeddings=True)
        for sample_name, sample_data in results.items():
            assert sample_data["Output"]["Embeddings1"] is not None
            assert sample_data["Output"]["Embeddings2"] is not None


@pytest.mark.slow
class TestPredictModelAttentions:

    def test_attentions_captured(self, tmp_path):
        from pathogenfinder2.dl.model import PathogenDLModel
        pf2, input_df = _create_embedding_file(tmp_path)
        model_dl = PathogenDLModel(
            model_parameters=pf2.pf2_config["Model Parameters"],
            misc_parameters=pf2.pf2_config["Misc Parameters"],
            seed=pf2.pf2_config["Model Parameters"]["Seed"],
        )
        results = model_dl.predict_model(input_metapath=input_df, save_attentions=True)
        for sample_name, sample_data in results.items():
            att = sample_data["Output"]["Attention"]
            assert att is not None
            assert len(att) > 0
