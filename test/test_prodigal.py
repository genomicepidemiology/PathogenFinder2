"""
Tests for ProdigalRunner.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.preprocessing.prodigal import ProdigalRunner


# ---------------------------------------------------------------------------
# count_proteins
# ---------------------------------------------------------------------------

class TestCountProteins:

    def test_counts_headers(self, tmp_path):
        fasta = tmp_path / "prots.faa"
        fasta.write_text(">prot1\nACDE\n>prot2\nMKLV\n>prot3\nWWWW\n")
        assert ProdigalRunner.count_proteins(str(fasta)) == 3

    def test_empty_fasta_returns_zero(self, tmp_path):
        fasta = tmp_path / "empty.faa"
        fasta.write_text("")
        assert ProdigalRunner.count_proteins(str(fasta)) == 0

    def test_ignores_sequence_lines(self, tmp_path):
        fasta = tmp_path / "prots.faa"
        fasta.write_text(">prot1\nACDE\nWWW\n")
        assert ProdigalRunner.count_proteins(str(fasta)) == 1


# ---------------------------------------------------------------------------
# __init__ defaults
# ---------------------------------------------------------------------------

class TestProdigalRunnerInit:

    def test_default_log_folder_is_output_folder(self, tmp_path):
        runner = ProdigalRunner(output_folder=str(tmp_path))
        assert runner.log_folder == str(tmp_path)

    def test_custom_log_folder(self, tmp_path):
        log = tmp_path / "logs"
        log.mkdir()
        runner = ProdigalRunner(output_folder=str(tmp_path), log_folder=str(log))
        assert runner.log_folder == str(log)

    def test_default_prodigal_path(self, tmp_path):
        runner = ProdigalRunner(output_folder=str(tmp_path))
        assert runner.prodigal_path == "prodigal"


# ---------------------------------------------------------------------------
# __call__ (mocked subprocess)
# ---------------------------------------------------------------------------

class TestProdigalRunnerCall:

    def _make_runner(self, tmp_path):
        return ProdigalRunner(output_folder=str(tmp_path), prodigal_path="prodigal")

    @patch("pathogenfinder2.preprocessing.prodigal.subprocess.Popen")
    def test_returns_six_values(self, mock_popen, tmp_path):
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        mock_popen.return_value = proc

        runner = self._make_runner(tmp_path)
        result = runner(input_filepath=str(tmp_path / "genome.fna"))
        assert len(result) == 6

    @patch("pathogenfinder2.preprocessing.prodigal.subprocess.Popen")
    def test_default_prot_file_path(self, mock_popen, tmp_path):
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        mock_popen.return_value = proc

        runner = self._make_runner(tmp_path)
        prot_file, *_ = runner(input_filepath=str(tmp_path / "g.fna"))
        assert prot_file == str(tmp_path / "PredictedProteins.faa")

    @patch("pathogenfinder2.preprocessing.prodigal.subprocess.Popen")
    def test_custom_prot_file(self, mock_popen, tmp_path):
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        mock_popen.return_value = proc

        runner = self._make_runner(tmp_path)
        custom = str(tmp_path / "custom.faa")
        prot_file, *_ = runner(input_filepath=str(tmp_path / "g.fna"),
                               prot_file=custom)
        assert prot_file == custom

    @patch("pathogenfinder2.preprocessing.prodigal.subprocess.Popen")
    def test_stdout_none_when_disabled(self, mock_popen, tmp_path):
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        mock_popen.return_value = proc

        runner = self._make_runner(tmp_path)
        _, gbk_file, _, _, _, _ = runner(
            input_filepath=str(tmp_path / "g.fna"), stdout=False
        )
        assert gbk_file is None

    @patch("pathogenfinder2.preprocessing.prodigal.subprocess.Popen")
    def test_stats_none_when_disabled(self, mock_popen, tmp_path):
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        mock_popen.return_value = proc

        runner = self._make_runner(tmp_path)
        _, _, stats_file, _, _, _ = runner(
            input_filepath=str(tmp_path / "g.fna"), stats=False
        )
        assert stats_file is None

    @patch("pathogenfinder2.preprocessing.prodigal.subprocess.Popen")
    def test_writes_log_files(self, mock_popen, tmp_path):
        proc = MagicMock()
        proc.communicate.return_value = (b"stdout content", b"stderr content")
        mock_popen.return_value = proc

        runner = self._make_runner(tmp_path)
        runner(input_filepath=str(tmp_path / "g.fna"))

        assert (tmp_path / "prodigal.out").read_bytes() == b"stdout content"
        assert (tmp_path / "prodigal.err").read_bytes() == b"stderr content"
