"""
Tests for SetupSwissProt — all network and subprocess calls are mocked.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.setup_prot_db import SetupSwissProt
from pathogenfinder2.exceptions import ExternalToolError


# ---------------------------------------------------------------------------
# Minimal stub GO DAG
# ---------------------------------------------------------------------------

def _make_go_dag(terms: dict):
    """
    Build a dict-like mock GO DAG.

    ``terms`` maps GO ID → (namespace, depth, parents_set).
    """
    nodes = {}
    for gid, (ns, depth, parents) in terms.items():
        node = MagicMock()
        node.namespace = ns
        node.depth = depth
        node.get_all_parents = MagicMock(return_value=parents)
        nodes[gid] = node

    dag = MagicMock()
    dag.__contains__ = lambda self, k: k in nodes
    dag.__getitem__ = lambda self, k: nodes[k]
    return dag


# ---------------------------------------------------------------------------
# check_metadata
# ---------------------------------------------------------------------------

class TestCheckMetadata:

    def test_valid_df_passes(self):
        df = pd.DataFrame({
            "Entry": ["A0A"],
            "Entry Name": ["GENE_ECOLI"],
            "Gene Ontology (biological process)": ["cell signaling [GO:0007165]"],
        })
        SetupSwissProt.check_metadata(df)  # should not raise

    def test_missing_entry_raises(self):
        df = pd.DataFrame({"Entry Name": ["x"], "Gene Ontology (biological process)": ["y"]})
        with pytest.raises(KeyError, match="Entry"):
            SetupSwissProt.check_metadata(df)

    def test_missing_entry_name_raises(self):
        df = pd.DataFrame({"Entry": ["A0A"], "Gene Ontology (biological process)": ["y"]})
        with pytest.raises(KeyError):
            SetupSwissProt.check_metadata(df)

    def test_missing_go_column_raises(self):
        df = pd.DataFrame({"Entry": ["A0A"], "Entry Name": ["x"]})
        with pytest.raises(KeyError):
            SetupSwissProt.check_metadata(df)


# ---------------------------------------------------------------------------
# bp_filter_and_deepest
# ---------------------------------------------------------------------------

class TestBpFilterAndDeepest:

    def _dag(self):
        return _make_go_dag({
            "GO:0001": ("biological_process", 3, {"GO:ROOT"}),
            "GO:0002": ("biological_process", 5, {"GO:0001", "GO:ROOT"}),
            "GO:0003": ("molecular_function", 4, set()),
            "GO:0004": ("biological_process", 2, {"GO:ROOT"}),
        })

    def test_filters_non_bp_terms(self):
        dag = self._dag()
        bp_ids, _ = SetupSwissProt.bp_filter_and_deepest(
            ["GO:0001", "GO:0003"], dag
        )
        assert "GO:0003" not in bp_ids
        assert "GO:0001" in bp_ids

    def test_deepest_returned(self):
        dag = self._dag()
        _, deepest = SetupSwissProt.bp_filter_and_deepest(
            ["GO:0001", "GO:0002"], dag
        )
        assert deepest == ["GO:0002"]

    def test_empty_input_returns_empty(self):
        dag = self._dag()
        bp_ids, deepest = SetupSwissProt.bp_filter_and_deepest([], dag)
        assert bp_ids == []
        assert deepest == []

    def test_unknown_terms_excluded(self):
        dag = self._dag()
        bp_ids, _ = SetupSwissProt.bp_filter_and_deepest(
            ["GO:9999", "GO:0001"], dag
        )
        assert "GO:9999" not in bp_ids

    def test_prune_ancestors_removes_ancestor(self):
        dag = self._dag()
        bp_ids, deepest = SetupSwissProt.bp_filter_and_deepest(
            ["GO:0001", "GO:0002"], dag, prune_ancestors=True
        )
        assert deepest == ["GO:0002"]

    def test_no_prune_keeps_all_bp(self):
        dag = self._dag()
        bp_ids, deepest = SetupSwissProt.bp_filter_and_deepest(
            ["GO:0001", "GO:0002"], dag, prune_ancestors=False
        )
        assert "GO:0001" in bp_ids
        assert "GO:0002" in bp_ids

    def test_keep_all_ties_returns_multiple(self):
        dag = _make_go_dag({
            "GO:0010": ("biological_process", 4, set()),
            "GO:0011": ("biological_process", 4, set()),
        })
        _, deepest = SetupSwissProt.bp_filter_and_deepest(
            ["GO:0010", "GO:0011"], dag,
            prune_ancestors=False, keep_all_ties=True
        )
        assert len(deepest) == 2


# ---------------------------------------------------------------------------
# set_goterm
# ---------------------------------------------------------------------------

class TestSetGoterm:

    def test_returns_series(self):
        dag = _make_go_dag({
            "GO:0001": ("biological_process", 3, set()),
        })
        df = pd.DataFrame({
            "Gene Ontology (biological process)": ["cell signaling [GO:0001]"]
        })
        result = SetupSwissProt.set_goterm(dag, df)
        assert isinstance(result, pd.Series)
        assert len(result) == 1

    def test_empty_go_string_yields_empty(self):
        dag = _make_go_dag({})
        df = pd.DataFrame({"Gene Ontology (biological process)": [""]})
        result = SetupSwissProt.set_goterm(dag, df)
        assert result.iloc[0] == ""

    def test_nan_yields_empty(self):
        dag = _make_go_dag({})
        df = pd.DataFrame({"Gene Ontology (biological process)": [float("nan")]})
        result = SetupSwissProt.set_goterm(dag, df)
        assert result.iloc[0] == ""

    def test_no_bp_term_yields_empty(self):
        dag = _make_go_dag({
            "GO:0003": ("molecular_function", 4, set()),
        })
        df = pd.DataFrame({
            "Gene Ontology (biological process)": ["catalytic activity [GO:0003]"]
        })
        result = SetupSwissProt.set_goterm(dag, df)
        assert result.iloc[0] == ""


# ---------------------------------------------------------------------------
# stream_with_progress
# ---------------------------------------------------------------------------

class TestStreamWithProgress:

    def test_writes_content_to_file(self, tmp_path):
        out = tmp_path / "data.bin"
        fake_resp = MagicMock()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {"Content-Length": "5"}
        fake_resp.iter_content = MagicMock(return_value=[b"hello"])

        with patch("pathogenfinder2.setup_prot_db.requests.get", return_value=fake_resp):
            SetupSwissProt.stream_with_progress(
                "http://fake.url", {}, str(out), desc="test"
            )
        assert out.read_bytes() == b"hello"


# ---------------------------------------------------------------------------
# download_go_basic
# ---------------------------------------------------------------------------

class TestDownloadGoBasic:

    def test_invalid_fmt_raises(self, tmp_path):
        with pytest.raises(ValueError, match="fmt must be"):
            SetupSwissProt.download_go_basic(str(tmp_path), fmt="xml")

    def test_creates_file(self, tmp_path):
        fake_resp = MagicMock()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.raise_for_status = MagicMock()
        fake_resp.iter_content = MagicMock(return_value=[b"obo content"])

        with patch("pathogenfinder2.setup_prot_db.requests.get", return_value=fake_resp):
            result = SetupSwissProt.download_go_basic(str(tmp_path))
        assert Path(result).exists()
        assert result.endswith(".obo")

    def test_returned_path_is_absolute(self, tmp_path):
        fake_resp = MagicMock()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.raise_for_status = MagicMock()
        fake_resp.iter_content = MagicMock(return_value=[b"content"])

        with patch("pathogenfinder2.setup_prot_db.requests.get", return_value=fake_resp):
            result = SetupSwissProt.download_go_basic(str(tmp_path))
        assert Path(result).is_absolute()


# ---------------------------------------------------------------------------
# download_swissprot_bacteria
# ---------------------------------------------------------------------------

class TestDownloadSwissprotBacteria:

    def test_calls_stream_twice(self, tmp_path):
        with patch.object(SetupSwissProt, "stream_with_progress") as mock_stream:
            SetupSwissProt.download_swissprot_bacteria(str(tmp_path))
        assert mock_stream.call_count == 2

    def test_returns_two_paths(self, tmp_path):
        with patch.object(SetupSwissProt, "stream_with_progress"):
            fasta, tsv = SetupSwissProt.download_swissprot_bacteria(str(tmp_path))
        assert fasta.endswith(".fasta")
        assert tsv.endswith(".tsv")


# ---------------------------------------------------------------------------
# diamond_index
# ---------------------------------------------------------------------------

class TestDiamondIndex:

    def test_calls_subprocess_run(self):
        with patch("pathogenfinder2.setup_prot_db.subprocess.run") as mock_run:
            result = SetupSwissProt.diamond_index("db.faa", "db", "diamond")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "makedb" in cmd

    def test_returns_dmnd_path(self):
        with patch("pathogenfinder2.setup_prot_db.subprocess.run"):
            result = SetupSwissProt.diamond_index("db.faa", "mydb", "diamond")
        assert result == "mydb.dmnd"

    def test_raises_external_tool_error_on_failure(self):
        err = subprocess.CalledProcessError(1, "diamond", stderr=b"error msg")
        with patch("pathogenfinder2.setup_prot_db.subprocess.run", side_effect=err):
            with pytest.raises(ExternalToolError, match="DIAMOND"):
                SetupSwissProt.diamond_index("db.faa", "db", "diamond")


# ---------------------------------------------------------------------------
# format_metadata (end-to-end with mocked GODag)
# ---------------------------------------------------------------------------

class TestFormatMetadata:

    def _make_tsv(self, tmp_path):
        tsv = tmp_path / "uniprot.tsv"
        tsv.write_text(
            "Entry\tEntry Name\tGene Ontology (biological process)\n"
            "A0A001\tGENE_ECOLI\tcell signaling [GO:0007165]\n"
            "B0B002\tGENE2_HUMAN\t\n"
        )
        return str(tsv)

    def test_creates_output_file(self, tmp_path):
        tsv = self._make_tsv(tmp_path)
        out = str(tmp_path / "out.tsv")
        dag = _make_go_dag({
            "GO:0007165": ("biological_process", 4, set()),
        })
        with patch("pathogenfinder2.setup_prot_db.GODag", return_value=dag):
            SetupSwissProt.format_metadata(tsv, "go.obo", out)
        assert Path(out).exists()

    def test_raises_if_output_exists(self, tmp_path):
        tsv = self._make_tsv(tmp_path)
        out = tmp_path / "out.tsv"
        out.touch()
        with pytest.raises(OSError):
            SetupSwissProt.format_metadata(tsv, "go.obo", str(out))

    def test_output_has_expected_columns(self, tmp_path):
        tsv = self._make_tsv(tmp_path)
        out = str(tmp_path / "out2.tsv")
        dag = _make_go_dag({
            "GO:0007165": ("biological_process", 4, set()),
        })
        with patch("pathogenfinder2.setup_prot_db.GODag", return_value=dag):
            SetupSwissProt.format_metadata(tsv, "go.obo", out)
        result = pd.read_csv(out, sep="\t")
        for col in ("Entry", "Entry ID", "GO Term"):
            assert col in result.columns

    def test_entry_id_derived_from_entry_name(self, tmp_path):
        tsv = self._make_tsv(tmp_path)
        out = str(tmp_path / "out3.tsv")
        dag = _make_go_dag({
            "GO:0007165": ("biological_process", 4, set()),
        })
        with patch("pathogenfinder2.setup_prot_db.GODag", return_value=dag):
            SetupSwissProt.format_metadata(tsv, "go.obo", out)
        result = pd.read_csv(out, sep="\t")
        # "GENE_ECOLI" → Entry ID = "GENE"
        assert result["Entry ID"].iloc[0] == "GENE"
