"""
Tests for pf2_arguments() CLI parser.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.cli import pf2_arguments, check_arguments
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(args):
    """Run the parser with a fixed argv, returning the Namespace."""
    with patch("sys.argv", ["pathogenfinder2"] + args):
        return pf2_arguments()


# ---------------------------------------------------------------------------
# check_arguments validation
# ---------------------------------------------------------------------------

class TestCheckArguments:

    def _ns(self, **kwargs):
        import argparse
        defaults = dict(action="Prediction", attProteins=None, dbProteins=None,
                        inputFile="in.fna", multipleFiles=None)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_error_when_valid(self):
        ns = self._ns()
        check_arguments(ns)  # should not raise

    def test_align_without_db_raises(self):
        ns = self._ns(attProteins="align", dbProteins=None)
        with pytest.raises(ValueError, match="--dbProteins"):
            check_arguments(ns)

    def test_align_with_db_passes(self):
        ns = self._ns(attProteins="align", dbProteins="db.dmnd")
        check_arguments(ns)  # should not raise

    def test_no_input_raises(self):
        ns = self._ns(inputFile=None, multipleFiles=None)
        with pytest.raises(ValueError, match="-i/--inputFile"):
            check_arguments(ns)

    def test_multiple_files_accepted(self):
        ns = self._ns(inputFile=None, multipleFiles="list.txt")
        check_arguments(ns)  # should not raise


# ---------------------------------------------------------------------------
# predict subparser
# ---------------------------------------------------------------------------

class TestPredictParser:

    def test_predict_sets_action(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.action == "Prediction"

    def test_output_folder_stored(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.outputFolder == "/tmp/out"

    def test_format_seq_genome(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.formatSeq == "genome"

    def test_format_seq_embeddings(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "embeddings", "-i", "seq.npz"])
        assert args.formatSeq == "embeddings"

    def test_input_file_stored(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.inputFile == "seq.fna"

    def test_cge_flag_default_false(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.cge is False

    def test_cge_flag_set(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna", "--cge"])
        assert args.cge is True

    def test_gsea_default_false(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.gsea is False

    def test_att_proteins_default_none(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna"])
        assert args.attProteins is None

    def test_att_proteins_report(self):
        args = _parse(["predict", "-o", "/tmp/out", "-f", "genome", "-i", "seq.fna",
                        "--attProteins", "report"])
        assert args.attProteins == "report"

    def test_invalid_format_raises(self):
        with pytest.raises(SystemExit):
            _parse(["predict", "-o", "/tmp/out", "-f", "invalid", "-i", "seq.fna"])

    def test_missing_required_output_raises(self):
        with pytest.raises(SystemExit):
            _parse(["predict", "-f", "genome", "-i", "seq.fna"])


# ---------------------------------------------------------------------------
# setup_gsea subparser
# ---------------------------------------------------------------------------

class TestSetupParser:

    def test_setup_sets_action(self):
        args = _parse(["setup_gsea", "--outputFolder", "/tmp/out"])
        assert args.action == "Setup_SwissProt"

    def test_swissprot_tsv_default(self):
        args = _parse(["setup_gsea", "--outputFolder", "/tmp/out"])
        assert args.swissprot_tsv is None

    def test_swissprot_tsv_set(self):
        args = _parse(["setup_gsea", "--outputFolder", "/tmp/out",
                        "--swissprot_tsv", "uniprot.tsv"])
        assert args.swissprot_tsv == "uniprot.tsv"


# ---------------------------------------------------------------------------
# infer_proteomeLM subparser
# ---------------------------------------------------------------------------

class TestInferParser:

    def test_infer_sets_action(self):
        args = _parse(["infer_proteomeLM", "-o", "/tmp/out", "-i", "seq.fna"])
        assert args.action == "Infer"

    def test_output_folder_stored(self):
        args = _parse(["infer_proteomeLM", "-o", "/tmp/out", "-i", "seq.fna"])
        assert args.outputFolder == "/tmp/out"


# ---------------------------------------------------------------------------
# train subparser
# ---------------------------------------------------------------------------

class TestTrainParser:

    def test_train_sets_action(self):
        args = _parse(["train", "-o", "/tmp/out"])
        assert args.action == "Train"


# ---------------------------------------------------------------------------
# --version on root parser
# ---------------------------------------------------------------------------

class TestVersionFlag:

    def test_version_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc:
            _parse(["--version"])
        assert exc.value.code == 0
