"""
Tests for ProteomeDataset, PhenotypeInteger, NormalizeData, and EmbeddingData.
"""
import os
import pytest
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.dl.utils.data import (
    ProteomeDataset, PhenotypeInteger, NormalizeData, EmbeddingData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedding_h5(tmp_path):
    """Write a minimal HDF5 embedding file and return its path."""
    n_prots = 5
    dim = ProteomeDataset.Dim_Embedding
    path = tmp_path / "embeddings.h5"
    embeddings = np.random.rand(n_prots, dim).astype(np.float32)
    names = np.array([f"prot{i}" for i in range(n_prots)], dtype=object)
    with h5py.File(path, "w") as hf:
        hf.attrs["Amount Embeddings"] = n_prots
        hf.create_dataset("Embeddings", data=embeddings)
        hf.create_dataset("Names", data=names)
    return path


@pytest.fixture
def labelled_csv(tmp_path, embedding_h5):
    """Write a TSV metadata file with a PathoPhenotype column."""
    df = pd.DataFrame({
        "File_Embedding": [str(embedding_h5)],
        "PathoPhenotype": ["Pathogenic"],
    })
    csv_path = tmp_path / "meta.tsv"
    df.to_csv(csv_path, sep="\t", index=False)
    return csv_path


@pytest.fixture
def prediction_csv(tmp_path, embedding_h5):
    """Write a TSV metadata file without a PathoPhenotype column (inference)."""
    df = pd.DataFrame({"File_Embedding": [str(embedding_h5)]})
    csv_path = tmp_path / "meta.tsv"
    df.to_csv(csv_path, sep="\t", index=False)
    return csv_path


# ---------------------------------------------------------------------------
# PhenotypeInteger
# ---------------------------------------------------------------------------

class TestPhenotypeInteger:

    def test_single_pathogenic(self):
        t = PhenotypeInteger("Single")
        sample = {"Label": "Pathogenic"}
        out = t(sample)
        assert out["Label"][0][0] == 1.0

    def test_single_non_pathogenic(self):
        t = PhenotypeInteger("Single")
        sample = {"Label": "No Pathogenic"}
        out = t(sample)
        assert out["Label"][0][0] == 0.0

    def test_dual_pathogenic(self):
        t = PhenotypeInteger("Dual")
        sample = {"Label": "Pathogenic"}
        out = t(sample)
        assert list(out["Label"][0]) == [1, 0]

    def test_none_label_gives_nan(self):
        t = PhenotypeInteger("Single")
        sample = {"Label": None}
        out = t(sample)
        assert np.isnan(out["Label"][0])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            PhenotypeInteger("Unknown")


# ---------------------------------------------------------------------------
# NormalizeData
# ---------------------------------------------------------------------------

class TestNormalizeData:

    def test_normalizes_correctly(self, tmp_path):
        mean = np.ones(4, dtype=np.float32) * 2.0
        std = np.ones(4, dtype=np.float32) * 0.5
        npz_path = tmp_path / "stats.npz"
        np.savez(npz_path, mean=mean, std=std)

        nd = NormalizeData(saved_data=str(npz_path))
        inp = np.array([3.0, 2.0, 1.0, 2.5], dtype=np.float32)
        sample = {"Input": inp}
        out = nd(sample)
        expected = ((inp - mean) / std).astype(np.float32)
        np.testing.assert_allclose(out["Input"], expected)

    def test_none_saved_data_stores_none(self):
        nd = NormalizeData(saved_data=None)
        assert nd.mean_vec is None
        assert nd.std_vec is None


# ---------------------------------------------------------------------------
# ProteomeDataset
# ---------------------------------------------------------------------------

class TestProteomeDataset:

    def test_len(self, labelled_csv):
        ds = ProteomeDataset(csv_file=str(labelled_csv))
        assert len(ds) == 1

    def test_getitem_returns_expected_keys(self, labelled_csv):
        ds = ProteomeDataset(csv_file=str(labelled_csv))
        sample = ds[0]
        for key in ("Input", "Label", "Protein Count", "Protein_IDs", "File_Name"):
            assert key in sample

    def test_input_shape(self, labelled_csv, embedding_h5):
        ds = ProteomeDataset(csv_file=str(labelled_csv))
        sample = ds[0]
        assert sample["Input"].shape[1] == ProteomeDataset.Dim_Embedding

    def test_prediction_dataset_flag_labelled(self, labelled_csv):
        ds = ProteomeDataset(csv_file=str(labelled_csv))
        assert not ds.prediction_dataset

    def test_prediction_dataset_flag_unlabelled(self, prediction_csv):
        ds = ProteomeDataset(csv_file=str(prediction_csv))
        assert ds.prediction_dataset

    def test_invalid_input_type_raises(self, labelled_csv):
        with pytest.raises(ValueError):
            ProteomeDataset(csv_file=str(labelled_csv), input_type="invalid")

    def test_label_is_none_for_prediction_dataset(self, prediction_csv):
        ds = ProteomeDataset(csv_file=str(prediction_csv))
        sample = ds[0]
        assert sample["Label"] is None

    def test_protein_count_matches_h5(self, labelled_csv):
        ds = ProteomeDataset(csv_file=str(labelled_csv))
        sample = ds[0]
        assert sample["Protein Count"][0] == 5


# ---------------------------------------------------------------------------
# EmbeddingData.create_dataset
# ---------------------------------------------------------------------------

class TestEmbeddingData:

    def test_create_dataset_returns_proteome_dataset(self, prediction_csv):
        ds = EmbeddingData.create_dataset(
            input_type="protein_embeddings",
            data_df=str(prediction_csv),
            data_type="prediction",
        )
        assert isinstance(ds, ProteomeDataset)

    def test_invalid_data_type_raises(self, prediction_csv):
        with pytest.raises(ValueError):
            EmbeddingData.create_dataset(
                input_type="protein_embeddings",
                data_df=str(prediction_csv),
                data_type="invalid",
            )

    def test_load_data_returns_dataloader(self, prediction_csv):
        from torch.utils.data import DataLoader
        ds = EmbeddingData.create_dataset(
            input_type="protein_embeddings",
            data_df=str(prediction_csv),
            data_type="prediction",
        )
        loader = EmbeddingData.load_data(ds, batch_size=1, num_workers=0, shuffle=False)
        assert isinstance(loader, DataLoader)

    def test_load_data_single_batch(self, prediction_csv):
        ds = EmbeddingData.create_dataset(
            input_type="protein_embeddings",
            data_df=str(prediction_csv),
            data_type="prediction",
        )
        loader = EmbeddingData.load_data(ds, batch_size=1, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        assert "Input" in batch
        assert batch["Input"].shape[0] == 1
