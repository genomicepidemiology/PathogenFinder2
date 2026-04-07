"""
Tests for ConfigurationPF2 and FilesModule.
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.utils.configuration import ConfigurationPF2, FilesModule
from pathogenfinder2.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# ConfigurationPF2
# ---------------------------------------------------------------------------

class TestConfigurationPF2:

    def test_default_config_loads(self):
        cfg = ConfigurationPF2(mode="Prediction")
        assert "Model Parameters" in cfg
        assert "Misc Parameters" in cfg

    def test_default_mode_is_set(self):
        cfg = ConfigurationPF2(mode="Prediction")
        assert cfg["Misc Parameters"]["Actions"] == "Prediction"

    def test_model_name(self):
        cfg = ConfigurationPF2(mode="Prediction")
        assert cfg["Model Parameters"]["Model Name"] == "ConvNext-AddAtt"

    def test_four_weight_files_appended(self):
        cfg = ConfigurationPF2(mode="Prediction")
        assert len(cfg["Model Parameters"]["Network Weights"]) == 4

    def test_weight_files_exist(self):
        cfg = ConfigurationPF2(mode="Prediction")
        for wp in cfg["Model Parameters"]["Network Weights"]:
            assert Path(wp).is_file(), f"Weight file missing: {wp}"

    def test_user_config_false_sentinel_uses_default(self):
        """Passing False (the default) must use the base config, not raise."""
        cfg = ConfigurationPF2(mode="Prediction", user_config=False)
        assert cfg["Model Parameters"]["Model Name"] == "ConvNext-AddAtt"

    def test_validation_error_on_bad_schema(self):
        """A user config missing required Train Parameters must raise ConfigurationError."""
        # Provide minimum structure so add_torch_functions doesn't KeyError,
        # but omit required schema fields (e.g. 'Epochs') so jsonschema fires.
        bad_config = {
            "Misc Parameters": {"Report Results": "file"},
            "Model Parameters": {
                "Model Name": "ConvNext-AddAtt",
                "Network Weights": [],
                "Batch Size": 16,
                "Mixed Precision": True,
                "Loss Function": "bcelogits",
            },
            "Train Parameters": {
                "Optimizer": "NAdam",
                "Loss Function": "BCEWithLogitsLoss",
                "Learning Rate": 1e-4,
                "Weight Decay": 1e-5,
                # "Epochs" intentionally missing — required by schema
                "Save Model": "best_epoch",
            },
        }
        with pytest.raises(ConfigurationError):
            ConfigurationPF2(mode="Train", user_config=bad_config)

    def test_get_bplfile_returns_path(self):
        cfg = ConfigurationPF2(mode="Prediction")
        bpl = cfg.get_bplfile()
        assert str(bpl).endswith(".npz")


# ---------------------------------------------------------------------------
# FilesModule
# ---------------------------------------------------------------------------

class TestFilesModule:

    def test_creates_base_folder(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        assert os.path.isdir(fm["base_folder"])

    def test_base_folder_name(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        assert "predictionPF2" in fm["base_folder"]

    def test_raises_on_existing_output(self, tmp_path):
        FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        with pytest.raises(OSError):
            FilesModule(outputFolder=str(tmp_path), mode="Prediction")

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError):
            FilesModule(outputFolder=str(tmp_path), mode="InvalidMode")

    def test_add_input_single_file(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        dummy = tmp_path / "seq.fna"
        dummy.write_text(">seq\nACGT\n")
        fm.add_input(input_file=str(dummy))
        assert "PF2" in fm["input_files"]

    def test_add_input_exclusive_options(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path / "p2"), mode="Prediction")
        with pytest.raises(ValueError):
            fm.add_input(input_file="a.fna", multi_file="list.txt")

    def test_create_nestedoutput_creates_dirs(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        dummy = tmp_path / "seq.fna"
        dummy.write_text(">seq\nACGT\n")
        fm.create_nestedoutput(format_seq="embeddings", input_file=str(dummy))
        assert "PF2" in fm["folders"]["results"]
        assert os.path.isdir(fm["folders"]["results"]["PF2"])

    def test_accessor_methods(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        dummy = tmp_path / "seq.fna"
        dummy.write_text(">seq\nACGT\n")
        fm.create_nestedoutput(format_seq="embeddings", input_file=str(dummy))
        assert fm.results_folder("PF2") == fm["folders"]["results"]["PF2"]
        assert fm.log_folder("PF2") == fm["folders"]["log"]["PF2"]
        assert fm.embedding_file("PF2") == fm["data_files"]["embedding_file"]["PF2"]

    def test_postprocess_folder_accessor(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        dummy = tmp_path / "seq.fna"
        dummy.write_text(">seq\nACGT\n")
        fm.create_nestedoutput(format_seq="embeddings", input_file=str(dummy))
        fm.postprocess_output()
        folder = fm.postprocess_folder("PF2")
        assert os.path.isdir(folder)

    def test_save_inputmetadata(self, tmp_path):
        fm = FilesModule(outputFolder=str(tmp_path), mode="Prediction")
        dummy = tmp_path / "seq.fna"
        dummy.write_text(">seq\nACGT\n")
        fm.create_nestedoutput(format_seq="embeddings", input_file=str(dummy))
        metadata = fm.save_inputmetadata()
        assert "Input_Files" in metadata.columns
        assert len(metadata) == 1
