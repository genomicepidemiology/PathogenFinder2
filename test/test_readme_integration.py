"""
README integration tests — run the exact commands from the documentation.

These tests reproduce what a new user does after cloning the repo.
They require Prodigal, ProtT5, and GPU. Mark: @pytest.mark.slow
"""
import json
import os
import subprocess
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
TEST_GENOME_1 = REPO_ROOT / "test" / "data" / "GCF_000014385.1_ASM1438v1_genomic.fna"
TEST_GENOME_2 = REPO_ROOT / "test" / "data" / "GCF_006493955.1_ASM649395v1_genomic.fna"


def _run_cli(args, timeout=600):
    """Run pathogenfinder2 CLI and return the CompletedProcess."""
    result = subprocess.run(
        [sys.executable, "-m", "pathogenfinder2"] + args,
        capture_output=True, text=True, timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    return result


def _find_predictions_tsv(out_dir):
    """Find predictions.tsv recursively under the output directory."""
    for p in Path(out_dir).rglob("predictions.tsv"):
        return p
    return None


@pytest.mark.slow
class TestReadmeSimplePrediction:

    def test_predict_genome_exits_zero(self, tmp_path):
        result = _run_cli([
            "predict", "-i", str(TEST_GENOME_1),
            "-f", "genome", "-o", str(tmp_path),
        ])
        assert result.returncode == 0, (
            f"predict failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_predictions_tsv_created(self, tmp_path):
        _run_cli(["predict", "-i", str(TEST_GENOME_1), "-f", "genome", "-o", str(tmp_path)])
        tsv = _find_predictions_tsv(tmp_path)
        assert tsv is not None, f"predictions.tsv not found under {tmp_path}"

    def test_prediction_mean_in_valid_range(self, tmp_path):
        _run_cli(["predict", "-i", str(TEST_GENOME_1), "-f", "genome", "-o", str(tmp_path)])
        tsv = _find_predictions_tsv(tmp_path)
        import pandas as pd
        df = pd.read_csv(tsv, sep="\t", comment="#")
        assert "Prediction Mean" in df.columns
        mean_val = df["Prediction Mean"].iloc[0]
        assert 0.0 <= mean_val <= 1.0, f"Prediction Mean {mean_val} out of range [0, 1]"

    def test_phenotype_column_present(self, tmp_path):
        _run_cli(["predict", "-i", str(TEST_GENOME_1), "-f", "genome", "-o", str(tmp_path)])
        tsv = _find_predictions_tsv(tmp_path)
        import pandas as pd
        df = pd.read_csv(tsv, sep="\t", comment="#")
        assert "Phenotype" in df.columns
        assert df["Phenotype"].iloc[0] in ["Human Pathogenic", "Human Non Pathogenic"]


@pytest.mark.slow
class TestSecondGenome:

    def test_second_genome_exits_zero(self, tmp_path):
        result = _run_cli([
            "predict", "-i", str(TEST_GENOME_2),
            "-f", "genome", "-o", str(tmp_path),
        ])
        assert result.returncode == 0, (
            f"predict failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_second_genome_predictions_valid(self, tmp_path):
        _run_cli(["predict", "-i", str(TEST_GENOME_2), "-f", "genome", "-o", str(tmp_path)])
        tsv = _find_predictions_tsv(tmp_path)
        assert tsv is not None
        import pandas as pd
        df = pd.read_csv(tsv, sep="\t", comment="#")
        mean_val = df["Prediction Mean"].iloc[0]
        assert 0.0 <= mean_val <= 1.0


@pytest.mark.slow
class TestCGEOutput:

    def test_cge_json_created(self, tmp_path):
        result = _run_cli([
            "predict", "-i", str(TEST_GENOME_1),
            "-f", "genome", "-o", str(tmp_path), "--cge",
        ])
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        cge_files = list(Path(tmp_path).rglob("cge_out.json"))
        assert len(cge_files) > 0, f"cge_out.json not found under {tmp_path}"

    def test_cge_json_valid_structure(self, tmp_path):
        _run_cli([
            "predict", "-i", str(TEST_GENOME_1), "-f", "genome",
            "-o", str(tmp_path), "--cge",
        ])
        cge_file = next(Path(tmp_path).rglob("cge_out.json"))
        with open(cge_file) as f:
            data = json.load(f)
        assert "phenotypes_ml" in data
        assert "software_name" in data
        assert data["software_name"] == "PathogenFinder2"


@pytest.mark.slow
class TestEmbeddingsFormat:

    def _run_genome_first(self, tmp_path):
        genome_out = tmp_path / "genome_run"
        _run_cli([
            "predict", "-i", str(TEST_GENOME_1), "-f", "genome",
            "-o", str(genome_out),
        ])
        h5_files = list(genome_out.rglob("*.h5"))
        assert len(h5_files) > 0, f"No .h5 embedding file found under {genome_out}"
        return h5_files[0]

    def test_embeddings_format_exits_zero(self, tmp_path):
        h5_path = self._run_genome_first(tmp_path)
        emb_out = tmp_path / "emb_run"
        result = _run_cli([
            "predict", "-i", str(h5_path), "-f", "embeddings",
            "-o", str(emb_out),
        ])
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_embeddings_predictions_valid(self, tmp_path):
        h5_path = self._run_genome_first(tmp_path)
        emb_out = tmp_path / "emb_run"
        _run_cli([
            "predict", "-i", str(h5_path), "-f", "embeddings",
            "-o", str(emb_out),
        ])
        tsv = _find_predictions_tsv(emb_out)
        assert tsv is not None
        import pandas as pd
        df = pd.read_csv(tsv, sep="\t", comment="#")
        mean_val = df["Prediction Mean"].iloc[0]
        assert 0.0 <= mean_val <= 1.0


# ---------------------------------------------------------------------------
# Help commands — documented in README, no GPU needed
# ---------------------------------------------------------------------------

class TestHelpCommands:

    def test_main_help(self):
        result = _run_cli(["-h"])
        assert result.returncode == 0

    def test_predict_help(self):
        result = _run_cli(["predict", "-h"])
        assert result.returncode == 0

    def test_train_help(self):
        result = _run_cli(["train", "-h"])
        assert result.returncode == 0

    def test_infer_help(self):
        result = _run_cli(["infer_proteomeLM", "-h"])
        assert result.returncode == 0

    def test_setup_gsea_help(self):
        result = _run_cli(["setup_gsea", "-h"])
        assert result.returncode == 0

    def test_version(self):
        result = _run_cli(["--version"])
        assert result.returncode == 0
        assert "0.7.0" in result.stdout


# ---------------------------------------------------------------------------
# infer_proteomeLM — documented in README
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestInferProteomeLM:

    def test_infer_exits_zero(self, tmp_path):
        result = _run_cli([
            "infer_proteomeLM",
            "-i", str(TEST_GENOME_1),
            "-o", str(tmp_path),
        ])
        assert result.returncode == 0, (
            f"infer failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_infer_creates_h5(self, tmp_path):
        _run_cli([
            "infer_proteomeLM",
            "-i", str(TEST_GENOME_1),
            "-o", str(tmp_path),
        ])
        h5_files = list(Path(tmp_path).rglob("*.h5"))
        assert len(h5_files) > 0, f"No .h5 file found under {tmp_path}"
