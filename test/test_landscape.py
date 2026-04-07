"""
Tests for MapEmbeddings — all UMAP calls are mocked so no GPU is required.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_REF = 20   # number of reference proteomes
DIM = 16     # embedding dimension


@pytest.fixture
def ref_npz(tmp_path):
    """Minimal reference BPL NPZ file."""
    rng = np.random.default_rng(0)
    path = tmp_path / "bpl.npz"
    np.savez(
        path,
        embedding=rng.random((N_REF, DIM)).astype(np.float32),
        name_refseq=np.array([f"org{i}_NC_{i:06d}" for i in range(N_REF)]),
        species_name=np.array([f"Species {i}" for i in range(N_REF)]),
        strain_name=np.array([f"Strain{i}" for i in range(N_REF)]),
        refseq_id=np.array([f"NC_{i:06d}" for i in range(N_REF)]),
        taxonomy_id=np.arange(N_REF, dtype=np.int32),
    )
    return str(path)


@pytest.fixture
def query_npz(tmp_path):
    """Minimal query embedding NPZ (single proteome)."""
    rng = np.random.default_rng(1)
    path = tmp_path / "query.npz"
    np.savez(path, embeddings_1=rng.random((1, DIM)).astype(np.float32))
    return str(path)


def _fake_umap_class(n_ref=N_REF):
    """Return a mock UMAP class whose instances produce 2-D coordinates."""
    rng = np.random.default_rng(42)
    ref_coords = rng.random((n_ref, 2)).astype(np.float32)

    instance = MagicMock()
    instance.fit.return_value = instance

    def _transform(x):
        n = x.shape[0] if hasattr(x, "shape") else 1
        return rng.random((n, 2)).astype(np.float32)

    instance.transform.side_effect = _transform
    # make fit return coords for train step
    instance.fit.side_effect = lambda x: instance
    # the class call returns the instance
    cls = MagicMock(return_value=instance)
    return cls, instance, ref_coords


# ---------------------------------------------------------------------------
# MapEmbeddings construction
# ---------------------------------------------------------------------------

class TestMapEmbeddingsInit:

    def test_out_folder_stored(self, tmp_path, ref_npz):
        umap_cls, _, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            me = MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)
        assert me.out_folder == str(tmp_path)

    def test_train_data_is_2d(self, tmp_path, ref_npz):
        umap_cls, _, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            me = MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)
        assert me.train_data.ndim == 2
        assert me.train_data.shape[1] == 2

    def test_fit_model_stored(self, tmp_path, ref_npz):
        umap_cls, instance, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            me = MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)
        assert me.fit_model is instance


# ---------------------------------------------------------------------------
# fit_test_data
# ---------------------------------------------------------------------------

class TestFitTestData:

    def _make_me(self, tmp_path, ref_npz):
        umap_cls, _, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            return MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)

    def test_returns_2d_array(self, tmp_path, ref_npz, query_npz):
        me = self._make_me(tmp_path, ref_npz)
        result = me.fit_test_data(query_npz)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_result_has_2_columns(self, tmp_path, ref_npz, query_npz):
        me = self._make_me(tmp_path, ref_npz)
        result = me.fit_test_data(query_npz)
        assert result.shape[1] == 2

    def test_transform_called_with_correct_shape(self, tmp_path, ref_npz, query_npz):
        umap_cls, instance, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            me = MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)
        me.fit_test_data(query_npz)
        # transform called once during init (train) and once for query
        assert instance.transform.call_count == 2


# ---------------------------------------------------------------------------
# knn
# ---------------------------------------------------------------------------

class TestKnn:

    def _make_me_with_fixed_train(self, tmp_path, ref_npz):
        """Create MapEmbeddings and fix train_data to known 2-D coords."""
        umap_cls, _, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            me = MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)
        # Replace with deterministic 2-D grid so knn is predictable
        rng = np.random.default_rng(7)
        me.train_data = rng.random((N_REF, 2)).astype(np.float32)
        return me

    def test_returns_dataframe_and_array(self, tmp_path, ref_npz):
        me = self._make_me_with_fixed_train(tmp_path, ref_npz)
        query = np.array([[0.5, 0.5]], dtype=np.float32)
        df, arr = me.knn(query, k=3)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(arr, np.ndarray)

    def test_dataframe_has_k_rows(self, tmp_path, ref_npz):
        me = self._make_me_with_fixed_train(tmp_path, ref_npz)
        query = np.array([[0.5, 0.5]], dtype=np.float32)
        df, _ = me.knn(query, k=5)
        assert len(df) == 5

    def test_dataframe_has_required_columns(self, tmp_path, ref_npz):
        me = self._make_me_with_fixed_train(tmp_path, ref_npz)
        query = np.array([[0.5, 0.5]], dtype=np.float32)
        df, _ = me.knn(query, k=3)
        for col in ("Names", "Species", "Strain", "RefSeq", "Taxonomy", "Distances"):
            assert col in df.columns

    def test_neighbour_tsv_written(self, tmp_path, ref_npz):
        me = self._make_me_with_fixed_train(tmp_path, ref_npz)
        query = np.array([[0.5, 0.5]], dtype=np.float32)
        me.knn(query, k=3)
        assert (tmp_path / "closeneighbors_bpl.tsv").exists()

    def test_k_defaults_to_10(self, tmp_path, ref_npz):
        me = self._make_me_with_fixed_train(tmp_path, ref_npz)
        query = np.array([[0.5, 0.5]], dtype=np.float32)
        df, _ = me.knn(query)
        assert len(df) == 10


# ---------------------------------------------------------------------------
# make_graph
# ---------------------------------------------------------------------------

class TestMakeGraph:

    def _make_me(self, tmp_path, ref_npz):
        umap_cls, _, _ = _fake_umap_class()
        with patch("pathogenfinder2.postprocessing.landscape.umap.UMAP", umap_cls):
            from pathogenfinder2.postprocessing.landscape import MapEmbeddings
            me = MapEmbeddings(out_folder=str(tmp_path), data_embed=ref_npz)
        me.train_data = np.random.default_rng(3).random((N_REF, 2)).astype(np.float32)
        return me

    def test_creates_png(self, tmp_path, ref_npz):
        import matplotlib
        matplotlib.use("Agg")
        me = self._make_me(tmp_path, ref_npz)
        test_coords = np.array([[0.3, 0.4]])
        closer = np.random.default_rng(5).random((3, 2)).astype(np.float32)
        me.make_graph(test_coords, closer)
        assert (tmp_path / "mapped_bpl.png").exists()

    def test_no_error_with_add_sp_false(self, tmp_path, ref_npz):
        import matplotlib
        matplotlib.use("Agg")
        me = self._make_me(tmp_path, ref_npz)
        test_coords = np.array([[0.3, 0.4]])
        closer = np.random.default_rng(5).random((3, 2)).astype(np.float32)
        me.make_graph(test_coords, closer, add_sp=False)  # should not raise
