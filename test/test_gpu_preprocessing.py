"""
GPU preprocessing tests — real Prodigal and ProtT5 runs.

Requires: Prodigal in PATH, ProtT5 model (downloaded on first run), GPU.
Mark: @pytest.mark.slow
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
TEST_GENOME = REPO_ROOT / "test" / "data" / "GCF_000014385.1_ASM1438v1_genomic.fna"


def _make_pf2(tmp_path):
    from pathogenfinder2.main import PathogenFinder2
    return PathogenFinder2(mode="Prediction", outPath=str(tmp_path))


@pytest.mark.slow
class TestPredictProteinContent:

    def test_returns_success(self, tmp_path):
        pf2 = _make_pf2(tmp_path)
        pf2.files_module.create_nestedoutput(
            format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
        result = pf2.predict_protein_content()
        base = list(result.keys())[0]
        assert result[base] == "Success"

    def test_protein_fasta_created(self, tmp_path):
        pf2 = _make_pf2(tmp_path)
        pf2.files_module.create_nestedoutput(
            format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
        pf2.predict_protein_content()
        faa_files = list(Path(tmp_path).rglob("*.faa"))
        assert len(faa_files) > 0, f"No .faa protein file found under {tmp_path}"

    def test_protein_count_positive(self, tmp_path):
        pf2 = _make_pf2(tmp_path)
        pf2.files_module.create_nestedoutput(
            format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
        pf2.predict_protein_content()
        faa_files = list(Path(tmp_path).rglob("*.faa"))
        with open(faa_files[0]) as f:
            count = sum(1 for line in f if line.startswith(">"))
        assert count > 0, "Prodigal produced 0 proteins"


@pytest.mark.slow
class TestInferEmbeddings:

    def _run_prodigal_first(self, tmp_path):
        pf2 = _make_pf2(tmp_path)
        pf2.files_module.create_nestedoutput(
            format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
        pf2.predict_protein_content()
        return pf2

    def test_h5_created(self, tmp_path):
        pf2 = self._run_prodigal_first(tmp_path)
        pf2.infer_embeddings()
        h5_files = list(Path(tmp_path).rglob("*.h5"))
        assert len(h5_files) > 0, f"No .h5 embedding file created under {tmp_path}"

    def test_h5_has_embeddings_dataset(self, tmp_path):
        import h5py
        pf2 = self._run_prodigal_first(tmp_path)
        pf2.infer_embeddings()
        h5_file = list(Path(tmp_path).rglob("*.h5"))[0]
        with h5py.File(h5_file, "r") as hf:
            assert "Embeddings" in hf
            assert "Names" in hf

    def test_embedding_dimension_is_1024(self, tmp_path):
        import h5py
        pf2 = self._run_prodigal_first(tmp_path)
        pf2.infer_embeddings()
        h5_file = list(Path(tmp_path).rglob("*.h5"))[0]
        with h5py.File(h5_file, "r") as hf:
            assert hf["Embeddings"].shape[1] == 1024


@pytest.mark.slow
class TestRunPreprocessingGenome:

    def test_genome_creates_prodigal_and_embeddings(self, tmp_path):
        pf2 = _make_pf2(tmp_path)
        pf2.files_module.create_nestedoutput(
            format_seq="genome", input_file=str(TEST_GENOME), multi_file=False)
        success = pf2._run_preprocessing(format_seq="genome")
        base = list(success.keys())[0]
        assert success[base] == "Success"
        assert len(list(Path(tmp_path).rglob("*.faa"))) > 0, "No protein FASTA"
        assert len(list(Path(tmp_path).rglob("*.h5"))) > 0, "No embedding file"
