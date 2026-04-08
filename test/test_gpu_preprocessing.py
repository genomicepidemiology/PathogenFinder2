"""
GPU preprocessing tests — real Prodigal and ProtT5 runs.
Uses session-scoped fixture from conftest.py.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.slow
class TestPredictProteinContent:

    def test_returns_success(self, gpu_preprocessed):
        pf2, _, out = gpu_preprocessed
        faa_files = list(Path(out).rglob("*.faa"))
        assert len(faa_files) > 0, "No .faa protein file found"

    def test_protein_count_positive(self, gpu_preprocessed):
        _, _, out = gpu_preprocessed
        faa_files = list(Path(out).rglob("*.faa"))
        with open(faa_files[0]) as f:
            count = sum(1 for line in f if line.startswith(">"))
        assert count > 0, "Prodigal produced 0 proteins"


@pytest.mark.slow
class TestInferEmbeddings:

    def test_h5_created(self, gpu_preprocessed):
        _, _, out = gpu_preprocessed
        h5_files = list(Path(out).rglob("*.h5"))
        assert len(h5_files) > 0, "No .h5 embedding file found"

    def test_h5_has_embeddings_dataset(self, gpu_preprocessed):
        import h5py
        _, _, out = gpu_preprocessed
        h5_file = list(Path(out).rglob("*.h5"))[0]
        with h5py.File(h5_file, "r") as hf:
            assert "Embeddings" in hf
            assert "Names" in hf

    def test_embedding_dimension_is_1024(self, gpu_preprocessed):
        import h5py
        _, _, out = gpu_preprocessed
        h5_file = list(Path(out).rglob("*.h5"))[0]
        with h5py.File(h5_file, "r") as hf:
            assert hf["Embeddings"].shape[1] == 1024
