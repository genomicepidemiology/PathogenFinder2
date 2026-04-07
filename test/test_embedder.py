"""
Tests for ProtT5Embedder and wrapper_multiple_core.

All model loading is mocked — no GPU or HuggingFace download required.
"""
import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model_and_vocab():
    """Return a (model, vocab) mock pair that produces fake embeddings."""
    vocab = MagicMock()
    vocab.return_value = {
        "input_ids": [[1, 2, 3, 4]],
        "attention_mask": [[1, 1, 1, 0]],
    }

    # model(input_ids, attention_mask) → obj with .last_hidden_state
    embedding_repr = MagicMock()
    embedding_repr.last_hidden_state = torch.zeros(1, 4, 32)
    model = MagicMock()
    model.return_value = embedding_repr
    model.__call__ = MagicMock(return_value=embedding_repr)
    model.float.return_value = model
    model.half.return_value = model
    model.to.return_value = model
    model.eval.return_value = model
    return model, vocab


def _make_embedder(pool_mode="mean"):
    """Create a ProtT5Embedder with all model I/O mocked."""
    from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
    model, vocab = _mock_model_and_vocab()
    with patch.object(ProtT5Embedder, "get_T5_model", return_value=(model, vocab)):
        emb = ProtT5Embedder(model_dir=None, pool_mode=pool_mode)
    emb.model = model
    emb.vocab = vocab
    return emb


# ---------------------------------------------------------------------------
# read_fasta (static, no model needed)
# ---------------------------------------------------------------------------

class TestReadFasta:

    def test_returns_dict(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "test.faa"
        f.write_text(">prot1\nACDE\n>prot2\nMKLV\n")
        result = ProtT5Embedder.read_fasta(str(f))
        assert isinstance(result, dict)

    def test_correct_number_of_entries(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "test.faa"
        f.write_text(">prot1\nACDE\n>prot2\nMKLV\n>prot3\nWWWW\n")
        result = ProtT5Embedder.read_fasta(str(f))
        assert len(result) == 3

    def test_sequence_uppercased(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "test.faa"
        f.write_text(">prot1\nacde\n")
        result = ProtT5Embedder.read_fasta(str(f))
        assert result["prot1"] == "ACDE"

    def test_gaps_removed(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "test.faa"
        f.write_text(">prot1\nAC-DE\n")
        result = ProtT5Embedder.read_fasta(str(f))
        assert result["prot1"] == "ACDE"

    def test_slash_replaced_in_header(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "test.faa"
        f.write_text(">gene/fragment\nACDE\n")
        result = ProtT5Embedder.read_fasta(str(f))
        assert "gene_fragment" in result

    def test_multiline_sequence_joined(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "test.faa"
        f.write_text(">prot1\nACDE\nMKLV\n")
        result = ProtT5Embedder.read_fasta(str(f))
        assert result["prot1"] == "ACDEMKLV"

    def test_empty_fasta_returns_empty_dict(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        f = tmp_path / "empty.faa"
        f.write_text("")
        result = ProtT5Embedder.read_fasta(str(f))
        assert result == {}


# ---------------------------------------------------------------------------
# ProtT5Embedder construction
# ---------------------------------------------------------------------------

class TestProtT5EmbedderInit:

    def test_pool_mode_mean(self):
        emb = _make_embedder(pool_mode="mean")
        assert emb.pool_mode == "mean"

    def test_pool_mode_max(self):
        emb = _make_embedder(pool_mode="max")
        assert emb.pool_mode == "max"

    def test_invalid_pool_mode_raises(self):
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder
        model, vocab = _mock_model_and_vocab()
        with patch.object(ProtT5Embedder, "get_T5_model", return_value=(model, vocab)):
            with pytest.raises(ValueError, match="pool mode"):
                ProtT5Embedder(model_dir=None, pool_mode="invalid")


# ---------------------------------------------------------------------------
# pool_embeddings
# ---------------------------------------------------------------------------

class TestPoolEmbeddings:

    def test_mean_pool_shape(self):
        emb = _make_embedder(pool_mode="mean")
        t = torch.randn(10, 32)  # seq_len=10, dim=32
        result = emb.pool_embeddings(t)
        assert result.shape == (32,)

    def test_max_pool_shape(self):
        emb = _make_embedder(pool_mode="max")
        t = torch.randn(10, 32)
        result = emb.pool_embeddings(t)
        assert result.shape == (32,)

    def test_mean_pool_value(self):
        emb = _make_embedder(pool_mode="mean")
        t = torch.ones(5, 4)
        result = emb.pool_embeddings(t)
        np.testing.assert_allclose(result, np.ones(4), atol=1e-5)

    def test_max_pool_picks_max(self):
        emb = _make_embedder(pool_mode="max")
        data = torch.tensor([[1.0, 2.0], [3.0, 0.5]])
        result = emb.pool_embeddings(data)
        assert result[0] == pytest.approx(3.0, abs=1e-5)
        assert result[1] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# get_embeddings (CPU, tiny FASTA, mocked model)
# ---------------------------------------------------------------------------

class TestGetEmbeddings:

    def test_creates_h5_file(self, tmp_path):
        fasta = tmp_path / "prots.faa"
        fasta.write_text(">p1\nACDE\n")
        h5_out = str(tmp_path / "emb.h5")

        emb = _make_embedder()
        # Make model return correct shape: batch=1, seq_len≥4, dim=32
        emb.vocab.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 0]],
        }
        fake_repr = MagicMock()
        fake_repr.last_hidden_state = torch.zeros(1, 5, 32)
        emb.model.return_value = fake_repr

        result = emb.get_embeddings(str(fasta), h5_out)
        assert result is True
        assert Path(h5_out).exists()

    def test_h5_contains_embeddings_dataset(self, tmp_path):
        import h5py
        fasta = tmp_path / "prots.faa"
        fasta.write_text(">p1\nACDE\n")
        h5_out = str(tmp_path / "emb.h5")

        emb = _make_embedder()
        emb.vocab.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 0]],
        }
        fake_repr = MagicMock()
        fake_repr.last_hidden_state = torch.zeros(1, 5, 32)
        emb.model.return_value = fake_repr

        emb.get_embeddings(str(fasta), h5_out)
        with h5py.File(h5_out, "r") as hf:
            assert "Embeddings" in hf
            assert "Names" in hf


# ---------------------------------------------------------------------------
# wrapper_multiple_core
# ---------------------------------------------------------------------------

class TestWrapperMultipleCore:

    def test_calls_get_embeddings_for_each_existing_file(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import wrapper_multiple_core

        fasta1 = tmp_path / "s1.faa"
        fasta2 = tmp_path / "s2.faa"
        fasta1.write_text(">p1\nACDE\n")
        fasta2.write_text(">p2\nMKLV\n")

        list_file = tmp_path / "files.txt"
        list_file.write_text(f"{fasta1}\n{fasta2}\n")

        out_dir = tmp_path / "emb_out"
        out_dir.mkdir()

        mock_emb = MagicMock()
        mock_emb.get_embeddings.return_value = True

        wrapper_multiple_core(mock_emb, str(list_file), str(out_dir),
                              pool_mode="mean", split_kmer=False)
        assert mock_emb.get_embeddings.call_count == 2

    def test_skips_missing_files(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import wrapper_multiple_core

        fasta1 = tmp_path / "real.faa"
        fasta1.write_text(">p1\nACDE\n")

        list_file = tmp_path / "files.txt"
        list_file.write_text(f"{fasta1}\n/nonexistent/ghost.faa\n")

        out_dir = tmp_path / "emb_out"
        out_dir.mkdir()

        mock_emb = MagicMock()
        wrapper_multiple_core(mock_emb, str(list_file), str(out_dir),
                              pool_mode="mean", split_kmer=False)
        assert mock_emb.get_embeddings.call_count == 1

    def test_overwrite_false_skips_existing_h5(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import wrapper_multiple_core

        fasta1 = tmp_path / "sample.faa"
        fasta1.write_text(">p1\nACDE\n")
        # pre-create the h5 file
        existing_h5 = tmp_path / "emb" / "sample.h5"
        existing_h5.parent.mkdir()
        existing_h5.touch()

        list_file = tmp_path / "files.txt"
        list_file.write_text(f"{fasta1}\n")

        mock_emb = MagicMock()
        wrapper_multiple_core(mock_emb, str(list_file), str(existing_h5.parent),
                              pool_mode="mean", split_kmer=False, overwrite=False)
        mock_emb.get_embeddings.assert_not_called()

    def test_overwrite_true_processes_existing_h5(self, tmp_path):
        from pathogenfinder2.preprocessing.embedder import wrapper_multiple_core

        fasta1 = tmp_path / "sample.faa"
        fasta1.write_text(">p1\nACDE\n")
        existing_h5 = tmp_path / "emb" / "sample.h5"
        existing_h5.parent.mkdir()
        existing_h5.touch()

        list_file = tmp_path / "files.txt"
        list_file.write_text(f"{fasta1}\n")

        mock_emb = MagicMock()
        wrapper_multiple_core(mock_emb, str(list_file), str(existing_h5.parent),
                              pool_mode="mean", split_kmer=False, overwrite=True)
        mock_emb.get_embeddings.assert_called_once()


# ---------------------------------------------------------------------------
# main() input validation
# ---------------------------------------------------------------------------

class TestEmbedderMain:

    def test_both_inputs_raises(self):
        from pathogenfinder2.preprocessing.embedder import main
        with pytest.raises(ValueError, match="cannot be used at the same time"):
            main(embed_out="/tmp/out.h5", input_seq="a.faa", input_txt="list.txt")

    def test_neither_input_raises(self):
        from pathogenfinder2.preprocessing.embedder import main
        with pytest.raises(ValueError, match="mutually required"):
            main(embed_out="/tmp/out.h5", input_seq=None, input_txt=None)
