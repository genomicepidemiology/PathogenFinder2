"""
Session-scoped fixtures for GPU integration tests.

Prodigal + ProtT5 run ONCE per test session. All GPU tests reuse the output.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
TEST_GENOME_1 = REPO_ROOT / "test" / "data" / "GCF_000014385.1_ASM1438v1_genomic.fna"
TEST_GENOME_2 = REPO_ROOT / "test" / "data" / "GCF_006493955.1_ASM649395v1_genomic.fna"


@pytest.fixture(scope="session")
def gpu_preprocessed(tmp_path_factory):
    """Run Prodigal + ProtT5 once, return (pf2, input_df_path) for reuse."""
    from pathogenfinder2.main import PathogenFinder2

    out = tmp_path_factory.mktemp("gpu_session")
    pf2 = PathogenFinder2(mode="Prediction", outPath=str(out))
    pf2.files_module.create_nestedoutput(
        format_seq="genome", input_file=str(TEST_GENOME_1), multi_file=False)
    pf2.predict_protein_content()
    pf2.infer_embeddings()
    input_df = pf2.files_module.save_inputmetadata()
    return pf2, input_df, out


@pytest.fixture(scope="session")
def gpu_network(gpu_preprocessed, tmp_path_factory):
    """Load a real NetworkModule with the first weight file."""
    from pathogenfinder2.dl.utils.network import NetworkModule
    from pathogenfinder2.dl.models.convnext_addatt import ConvNext_AddAtt_Net
    from pathogenfinder2.dl.utils.data import EmbeddingData

    pf2, input_df, _ = gpu_preprocessed
    cfg = pf2.pf2_config
    mp = cfg["Model Parameters"]

    net_dir = tmp_path_factory.mktemp("gpu_network")
    nm = NetworkModule(
        model_type=ConvNext_AddAtt_Net,
        model_parameters=mp,
        out_folder=str(net_dir),
        mixed_precision=mp["Mixed Precision"],
        loss_type=mp["Loss Function"],
    )
    nm.load_model(str(mp["Network Weights"][0]))

    dataset = EmbeddingData.create_dataset(
        input_type="protein_embeddings",
        data_df=input_df,
        data_type="prediction",
        dual_pred=False,
    )
    loader = EmbeddingData.load_data(
        dataset, batch_size=mp["Batch Size"],
        num_workers=mp["Data Parameters"]["num_workers"],
        asynchronity=mp["Data Parameters"]["asynchronity"],
    )
    return nm, loader, cfg
