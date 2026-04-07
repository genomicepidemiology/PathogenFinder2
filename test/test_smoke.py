"""
Smoke test for PathogenFinder2.

Verifies that the package can be imported, the main class can be instantiated,
and the default configuration loads correctly (model weights, architecture
parameters).  Does NOT run the full prediction pipeline (requires Prodigal,
ProtT5, and GPU), so it is safe to run in any CI environment.
"""
import pytest
from pathlib import Path

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
FNA_FILE = DATA_DIR / "GCF_000014385.1_ASM1438v1_genomic.fna"


def test_import():
    from pathogenfinder2 import PathogenFinder2  # noqa: F401


def test_default_config_loads():
    from pathogenfinder2.utils.configuration import ConfigurationPF2

    cfg = ConfigurationPF2(mode="Prediction")
    assert "Model Parameters" in cfg
    assert "Misc Parameters" in cfg
    assert cfg["Model Parameters"]["Model Name"] == "ConvNext-AddAtt"
    assert len(cfg["Model Parameters"]["Network Weights"]) == 4


def test_model_weights_exist():
    from pathogenfinder2.utils.configuration import ConfigurationPF2

    cfg = ConfigurationPF2(mode="Prediction")
    for weight_path in cfg["Model Parameters"]["Network Weights"]:
        assert Path(weight_path).is_file(), f"Weight file not found: {weight_path}"


def test_network_module_init():
    """Verify the network can be instantiated from the default config."""
    from pathogenfinder2.utils.configuration import ConfigurationPF2
    from pathogenfinder2.dl.model import PathogenDLModel

    cfg = ConfigurationPF2(mode="Prediction")
    cfg["Misc Parameters"]["Results Folder"] = str(TEST_DIR)
    cfg["Misc Parameters"]["Report Results"] = "file"
    cfg["Misc Parameters"]["Actions"] = "Prediction"
    cfg["Model Parameters"]["Memory Report"] = False

    model = PathogenDLModel(
        model_parameters=cfg["Model Parameters"],
        misc_parameters=cfg["Misc Parameters"],
    )
    assert model is not None


def test_fna_files_present():
    for fna in DATA_DIR.glob("*.fna"):
        assert fna.stat().st_size > 0, f"FNA file is empty: {fna}"
