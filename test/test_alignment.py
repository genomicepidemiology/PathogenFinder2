"""
Tests for MapProteins pure functions (no Diamond or GSEA required).
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.postprocessing.alignment import MapProteins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_att_npz(tmp_path):
    """Write a minimal attention NPZ file (4 attention heads, 5 proteins)."""
    n = 5
    att_path = tmp_path / "attentions.npz"
    np.savez(
        att_path,
        protIDs=np.array([f"prot{i} # desc{i}".encode() for i in range(n)]),
        attentions=np.random.rand(4, n).astype(np.float32),
    )
    return att_path


@pytest.fixture
def minimal_fasta(tmp_path):
    """Write a minimal protein FASTA file."""
    fasta_path = tmp_path / "proteins.faa"
    fasta_path.write_text(
        ">prot0 # desc0\nACDE\n>prot1 # desc1\nMKLV\n>prot2 # desc2\nWWWW\n"
    )
    return fasta_path


@pytest.fixture
def minimal_aln_df():
    """Minimal Diamond alignment result DataFrame."""
    return pd.DataFrame({
        "qseqid": ["prot0", "prot1", "prot2"],
        "sseqid": ["sp|A0A|GENE_ECOLI", "sp|B0B|GENE2_HUMAN", "sp|C0C|GENE3_YEAST"],
        "pident": [98.0, 75.0, 45.0],
        "length": [100, 80, 50],
        "mismatch": [2, 5, 10],
        "gapopen": [0, 1, 2],
        "qstart": [1, 1, 1],
        "qend": [100, 80, 50],
        "sstart": [1, 1, 1],
        "send": [100, 80, 50],
        "evalue": [1e-50, 1e-20, 1e-5],
        "bitscore": [200.0, 100.0, 50.0],
        "qtitle": ["q0", "q1", "q2"],
        "stitle": ["Protein A Tax=Ecoli TaxID=562 species", "Protein B Tax=Human TaxID=9606 species", "No Match Found"],
        "scovhsp": [100.0, 90.0, 50.0],
        "slen": [100, 90, 60],
        "Entry": ["A0A", "B0B", "C0C"],
    })


# ---------------------------------------------------------------------------
# read_protfasta / write_protfasta
# ---------------------------------------------------------------------------

class TestReadWriteProtFasta:

    def test_read_returns_dict(self, minimal_fasta):
        result = MapProteins.read_protfasta(str(minimal_fasta))
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_read_header_stripped(self, minimal_fasta):
        result = MapProteins.read_protfasta(str(minimal_fasta))
        assert "prot0" in result

    def test_read_sequence_content(self, minimal_fasta):
        result = MapProteins.read_protfasta(str(minimal_fasta))
        assert result["prot0"] == "ACDE"

    def test_roundtrip(self, tmp_path):
        original = {"protA": "ACDEFGHIKLM", "protB": "NOPQRSTVWY"}
        out_path = str(tmp_path / "out.faa")
        MapProteins.write_protfasta(out_path, original)
        recovered = MapProteins.read_protfasta(out_path)
        assert recovered == original

    def test_write_creates_file(self, tmp_path):
        out_path = str(tmp_path / "prots.faa")
        MapProteins.write_protfasta(out_path, {"p1": "ACDE"})
        assert Path(out_path).exists()


# ---------------------------------------------------------------------------
# check_metadata
# ---------------------------------------------------------------------------

class TestCheckMetadata:

    def test_valid_metadata_passes(self):
        df = pd.DataFrame({"Entry": ["A0A", "B0B"], "other": [1, 2]})
        MapProteins.check_metadata(df)  # should not raise

    def test_missing_entry_column_raises(self):
        df = pd.DataFrame({"WrongColumn": ["A0A"]})
        with pytest.raises(KeyError):
            MapProteins.check_metadata(df)


# ---------------------------------------------------------------------------
# load_attfile
# ---------------------------------------------------------------------------

class TestLoadAttfile:

    def test_returns_dataframe(self, minimal_att_npz):
        df = MapProteins.load_attfile(str(minimal_att_npz))
        assert isinstance(df, pd.DataFrame)

    def test_has_top_column(self, minimal_att_npz):
        df = MapProteins.load_attfile(str(minimal_att_npz))
        assert "top" in df.columns

    def test_top_count_default(self, minimal_att_npz):
        df = MapProteins.load_attfile(str(minimal_att_npz), top_proteins=3)
        assert df["top"].sum() == 3

    def test_top_count_capped_at_n(self, minimal_att_npz):
        # 5 proteins but asking for 10 — should give all 5
        df = MapProteins.load_attfile(str(minimal_att_npz), top_proteins=10)
        assert df["top"].sum() == 5

    def test_attmax_column_present_combine_max(self, minimal_att_npz):
        df = MapProteins.load_attfile(str(minimal_att_npz), combine_max=True)
        assert "attmax" in df.columns

    def test_protids_decoded(self, minimal_att_npz):
        df = MapProteins.load_attfile(str(minimal_att_npz))
        assert df["protIDs"].str.startswith("prot").all()


# ---------------------------------------------------------------------------
# rank_proteins
# ---------------------------------------------------------------------------

class TestRankProteins:

    def test_returns_series(self, minimal_att_npz):
        att_df = MapProteins.load_attfile(str(minimal_att_npz))
        aln_df = pd.DataFrame({
            "qseqid": att_df["protIDs"].tolist(),
            "sseqid": [f"sp|X{i}|GENE_{i}" for i in range(len(att_df))],
            "stitle": [f"Protein {i}" for i in range(len(att_df))],
            "attmax": att_df["attmax"].tolist(),
        })
        result = MapProteins.rank_proteins(aln_df)
        assert isinstance(result, pd.Series)

    def test_sorted_descending(self, minimal_att_npz):
        att_df = MapProteins.load_attfile(str(minimal_att_npz))
        aln_df = pd.DataFrame({
            "qseqid": att_df["protIDs"].tolist(),
            "sseqid": [f"sp|X{i}|GENE" for i in range(len(att_df))],
            "stitle": [f"P {i}" for i in range(len(att_df))],
            "attmax": att_df["attmax"].tolist(),
        })
        result = MapProteins.rank_proteins(aln_df)
        assert result.is_monotonic_decreasing


# ---------------------------------------------------------------------------
# create_proteinsets
# ---------------------------------------------------------------------------

class TestCreateProteinsets:

    def _make_gsea_df(self):
        return pd.DataFrame({
            "qseqid": ["p1", "p2", "p3", "p4"],
            "GO Term": ["GO:0001; GO:0002", "GO:0001", "GO:0003", None],
            "Entry ID": ["entry1", "entry2", "entry3", None],
            "attmax": [0.9, 0.8, 0.7, 0.6],
            "stitle": ["A", "B", "C", "D"],
        })

    def test_returns_dict(self):
        df = self._make_gsea_df()
        result = MapProteins.create_proteinsets(df, terms="Entry ID")
        assert isinstance(result, dict)

    def test_go_term_split_on_semicolon(self):
        df = self._make_gsea_df()
        result = MapProteins.create_proteinsets(df, terms="GO Term")
        assert "GO:0001" in result
        assert "GO:0002" in result

    def test_genes_deduplicated(self):
        df = pd.DataFrame({
            "qseqid": ["p1", "p1", "p2"],
            "GO Term": ["GO:0001", "GO:0001", "GO:0001"],
        })
        result = MapProteins.create_proteinsets(df, terms="GO Term")
        assert result["GO:0001"].count("p1") == 1

    def test_none_terms_excluded(self):
        df = self._make_gsea_df()
        result = MapProteins.create_proteinsets(df, terms="Entry ID")
        # None entry should not appear as a key
        assert None not in result


# ---------------------------------------------------------------------------
# MapProteins.__init__
# ---------------------------------------------------------------------------

class TestMapProteinsInit:

    def test_default_log_folder_is_folder_out(self, tmp_path):
        mp = MapProteins(folder_out=str(tmp_path), db_path="db.dmnd", diamond_path="diamond")
        assert mp.log_folder == str(tmp_path)

    def test_default_folder_tmp_is_folder_out(self, tmp_path):
        mp = MapProteins(folder_out=str(tmp_path), db_path="db.dmnd", diamond_path="diamond")
        assert mp.folder_tmp == str(tmp_path)

    def test_custom_log_folder(self, tmp_path):
        log = str(tmp_path / "log")
        mp = MapProteins(folder_out=str(tmp_path), db_path="x", diamond_path="d",
                         log_folder=log)
        assert mp.log_folder == log

    def test_metadata_loaded_when_provided(self, tmp_path):
        meta = tmp_path / "meta.tsv"
        meta.write_text("Entry\tName\nA0A\tGene1\n")
        mp = MapProteins(folder_out=str(tmp_path), db_path="x", diamond_path="d",
                         metadata=str(meta))
        assert mp.metadata is not None
        assert "Entry" in mp.metadata.columns

    def test_metadata_none_when_not_provided(self, tmp_path):
        mp = MapProteins(folder_out=str(tmp_path), db_path="x", diamond_path="d")
        assert mp.metadata is None
