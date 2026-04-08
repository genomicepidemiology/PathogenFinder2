"""
GPU landscape tests — real UMAP model, no mocks.

Requires: umap-learn, pynndescent<=0.5.13, the shipped umap_model.pkl.
Mark: @pytest.mark.slow
"""
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
BPL_DIR = REPO_ROOT / "src" / "pathogenfinder2" / "data" / "bpl"


def _get_bpl_paths():
    embeddings_npz = BPL_DIR / "embeddings.npz"
    umap_pkl = BPL_DIR / "umap_model.pkl"
    coords_npz = BPL_DIR / "bpl_coordinates.npz"
    return embeddings_npz, umap_pkl, coords_npz


@pytest.mark.slow
class TestMapEmbeddingsReal:

    def test_loads_with_prefitted_model(self, tmp_path):
        import joblib
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings
        emb_npz, umap_pkl, coords_npz = _get_bpl_paths()
        if not umap_pkl.exists():
            pytest.skip("umap_model.pkl not found")
        fitted_model = joblib.load(umap_pkl)
        train_data = np.load(coords_npz)["train_umap"]
        me = MapEmbeddings(
            out_folder=str(tmp_path), data_embed=str(emb_npz),
            fitted_model=fitted_model, train_data=train_data,
        )
        assert me.train_data.shape == (17035, 2)
        assert me.fit_model is fitted_model

    def test_fitdata_from_scratch(self, tmp_path):
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings
        emb_npz, _, _ = _get_bpl_paths()
        me = MapEmbeddings(out_folder=str(tmp_path), data_embed=str(emb_npz))
        assert me.train_data.shape[0] == 17035
        assert me.train_data.shape[1] == 2


@pytest.mark.slow
class TestFitTestDataReal:

    def test_transform_real_query(self, tmp_path):
        import joblib
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings
        emb_npz, umap_pkl, coords_npz = _get_bpl_paths()
        if not umap_pkl.exists():
            pytest.skip("umap_model.pkl not found")
        fitted_model = joblib.load(umap_pkl)
        train_data = np.load(coords_npz)["train_umap"]
        me = MapEmbeddings(
            out_folder=str(tmp_path), data_embed=str(emb_npz),
            fitted_model=fitted_model, train_data=train_data,
        )
        query_path = tmp_path / "query.npz"
        rng = np.random.default_rng(42)
        np.savez(query_path, embeddings_1=rng.random((1, 512)).astype(np.float32))
        result = me.fit_test_data(str(query_path))
        assert result.shape == (1, 2), f"Expected (1, 2), got {result.shape}"


@pytest.mark.slow
class TestKnnReal:

    def test_returns_10_neighbors(self, tmp_path):
        import joblib
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings
        emb_npz, umap_pkl, coords_npz = _get_bpl_paths()
        if not umap_pkl.exists():
            pytest.skip("umap_model.pkl not found")
        fitted_model = joblib.load(umap_pkl)
        train_data = np.load(coords_npz)["train_umap"]
        me = MapEmbeddings(
            out_folder=str(tmp_path), data_embed=str(emb_npz),
            fitted_model=fitted_model, train_data=train_data,
        )
        query_point = train_data.mean(axis=0, keepdims=True)
        df, arr = me.knn(query_point, k=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "Distances" in df.columns
        assert all(df["Distances"] >= 0)

    def test_tsv_written(self, tmp_path):
        import joblib
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings
        emb_npz, umap_pkl, coords_npz = _get_bpl_paths()
        if not umap_pkl.exists():
            pytest.skip("umap_model.pkl not found")
        fitted_model = joblib.load(umap_pkl)
        train_data = np.load(coords_npz)["train_umap"]
        me = MapEmbeddings(
            out_folder=str(tmp_path), data_embed=str(emb_npz),
            fitted_model=fitted_model, train_data=train_data,
        )
        query_point = train_data.mean(axis=0, keepdims=True)
        me.knn(query_point, k=5)
        assert (tmp_path / "closeneighbors_bpl.tsv").exists()


@pytest.mark.slow
class TestMakeGraphReal:

    def test_png_created(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import joblib
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings
        emb_npz, umap_pkl, coords_npz = _get_bpl_paths()
        if not umap_pkl.exists():
            pytest.skip("umap_model.pkl not found")
        fitted_model = joblib.load(umap_pkl)
        train_data = np.load(coords_npz)["train_umap"]
        me = MapEmbeddings(
            out_folder=str(tmp_path), data_embed=str(emb_npz),
            fitted_model=fitted_model, train_data=train_data,
        )
        query_point = train_data.mean(axis=0, keepdims=True)
        _, closer_arr = me.knn(query_point, k=5)
        me.make_graph(test_data=query_point, closer_data=closer_arr)
        png = tmp_path / "mapped_bpl.png"
        assert png.exists(), "mapped_bpl.png not created"
        assert png.stat().st_size > 0, "mapped_bpl.png is empty"
