"""
Tests for utils/io.py helper functions.
"""
import pytest
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.utils.io import read_multifiles, center_print


class TestReadMultifiles:

    def test_returns_two_lists(self, tmp_path):
        f = tmp_path / "files.txt"
        f.write_text("/path/to/sample1.fna\n/path/to/sample2.fna\n")
        paths, basenames = read_multifiles(str(f))
        assert isinstance(paths, list) and isinstance(basenames, list)

    def test_correct_number_of_entries(self, tmp_path):
        f = tmp_path / "files.txt"
        f.write_text("a.fna\nb.fna\nc.fna\n")
        paths, basenames = read_multifiles(str(f))
        assert len(paths) == 3 and len(basenames) == 3

    def test_paths_match_file_content(self, tmp_path):
        f = tmp_path / "files.txt"
        f.write_text("/data/sample1.fna\n/data/sample2.fna\n")
        paths, _ = read_multifiles(str(f))
        assert paths[0] == "/data/sample1.fna"
        assert paths[1] == "/data/sample2.fna"

    def test_basenames_extracted(self, tmp_path):
        f = tmp_path / "files.txt"
        f.write_text("/some/long/path/genome.fna\n")
        _, basenames = read_multifiles(str(f))
        assert basenames[0] == "genome.fna"

    def test_trailing_newline_stripped(self, tmp_path):
        f = tmp_path / "files.txt"
        f.write_text("seq.fna\n")
        paths, _ = read_multifiles(str(f))
        assert paths[0] == "seq.fna"

    def test_empty_file_returns_empty_lists(self, tmp_path):
        f = tmp_path / "files.txt"
        f.write_text("")
        paths, basenames = read_multifiles(str(f))
        assert paths == [] and basenames == []


class TestCenterPrint:

    def test_returns_string(self):
        result = center_print("hello")
        assert isinstance(result, str)

    def test_contains_message(self):
        result = center_print("hello")
        assert "hello" in result
