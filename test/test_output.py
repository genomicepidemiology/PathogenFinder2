"""
Tests for PredictionReport and CGEResults.
"""
import json
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.utils.output import PredictionReport, CGEResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ensemble_results(n_models=4):
    """Build a minimal ensemble_results dict as produced by predict_model."""
    preds = [round(0.6 + i * 0.01, 2) for i in range(n_models)]
    return {
        "sample1": {
            "Output": {"Prediction": preds, "Attention": [None],
                       "Embeddings1": [None], "Embeddings2": [None]},
            "Features": {"Filename": ["sample1.h5"], "ProtIDs": [],
                         "Proteome Length": 100, "Filepath": "sample1.h5"},
        }
    }


# ---------------------------------------------------------------------------
# PredictionReport.get_predictions
# ---------------------------------------------------------------------------

class TestGetPredictions:

    def test_adds_ensemble_predictions_key(self):
        results = _make_ensemble_results()
        PredictionReport.get_predictions(results)
        assert "Ensemble Predictions" in results["sample1"]

    def test_prediction_mean_correct(self):
        results = _make_ensemble_results(n_models=4)
        PredictionReport.get_predictions(results)
        ep = results["sample1"]["Ensemble Predictions"]
        expected_mean = np.mean(results["sample1"]["Output"]["Prediction"])
        assert abs(ep["Prediction Mean"] - expected_mean) < 1e-6

    def test_pathogenic_label_above_threshold(self):
        results = {"s": {"Output": {"Prediction": [0.8, 0.9, 0.7, 0.85]},
                         "Features": {"Filename": ["s"], "ProtIDs": [],
                                      "Proteome Length": 10, "Filepath": "s"}}}
        PredictionReport.get_predictions(results)
        assert results["s"]["Ensemble Predictions"]["Binary Prediction Mean"] == 1
        assert results["s"]["Ensemble Predictions"]["Phenotype"] == "Human Pathogenic"

    def test_non_pathogenic_label_below_threshold(self):
        results = {"s": {"Output": {"Prediction": [0.1, 0.2, 0.15, 0.12]},
                         "Features": {"Filename": ["s"], "ProtIDs": [],
                                      "Proteome Length": 10, "Filepath": "s"}}}
        PredictionReport.get_predictions(results)
        assert results["s"]["Ensemble Predictions"]["Binary Prediction Mean"] == 0
        assert results["s"]["Ensemble Predictions"]["Phenotype"] == "Human Non Pathogenic"

    def test_returns_none(self):
        """get_predictions must be in-place and return None."""
        results = _make_ensemble_results()
        retval = PredictionReport.get_predictions(results)
        assert retval is None

    def test_std_is_non_negative(self):
        results = _make_ensemble_results()
        PredictionReport.get_predictions(results)
        assert results["sample1"]["Ensemble Predictions"]["Prediction STD"] >= 0


# ---------------------------------------------------------------------------
# PredictionReport.save_report
# ---------------------------------------------------------------------------

class TestSaveReport:

    def _make_results_with_ensemble(self):
        results = _make_ensemble_results()
        PredictionReport.get_predictions(results)
        return results

    def test_saves_predictions_tsv(self, tmp_path):
        results = self._make_results_with_ensemble()
        out_folder = {"sample1": str(tmp_path)}
        report = PredictionReport(out_folder=out_folder)
        report.save_report(results_ensemble=results,
                           save_attentions=False, save_embeddings=False)
        assert (tmp_path / "predictions.tsv").exists()

    def test_tsv_contains_header_comment(self, tmp_path):
        results = self._make_results_with_ensemble()
        out_folder = {"sample1": str(tmp_path)}
        report = PredictionReport(out_folder=out_folder)
        report.save_report(results_ensemble=results,
                           save_attentions=False, save_embeddings=False)
        content = (tmp_path / "predictions.tsv").read_text()
        assert content.startswith("# Results From PathogenFinder2")


# ---------------------------------------------------------------------------
# CGEResults
# ---------------------------------------------------------------------------

class TestCGEResults:

    def test_instantiates(self):
        cge = CGEResults(args_dict={})
        assert cge.software_result["type"] == "software_result"
        assert cge.software_result["software_name"] == "PathogenFinder2"

    def test_run_id_is_sha1_hex(self):
        cge = CGEResults(args_dict={})
        run_id = cge.software_result["run_id"]
        assert len(run_id) == 40
        assert all(c in "0123456789abcdef" for c in run_id)

    def test_add_database(self):
        cge = CGEResults(args_dict={})
        db = cge.add_database(name="swissprot", version="2024_01")
        assert db["database_name"] == "swissprot"
        assert "swissprot_2024_01" in cge.software_result["databases"]

    def test_add_software_exec(self):
        cge = CGEResults(args_dict={})
        cge.add_software_exec(parameters={"input": "genome.fna"})
        assert len(cge.software_result["software_executions"]) == 1

    def test_add_log(self):
        cge = CGEResults(args_dict={})
        cge.add_log(result_summary="Success", log="All OK")
        assert cge.software_result["software_log"] == "All OK"
        assert cge.software_result["result_summary"] == "Success"

    def test_save_results_writes_valid_json(self, tmp_path):
        cge = CGEResults(args_dict={})
        out = tmp_path / "result.json"
        cge.save_results(str(out))
        with open(out) as f:
            data = json.load(f)
        assert data["type"] == "software_result"

    def test_generate_random_str_is_unique(self):
        existing = {}
        k1 = CGEResults.generate_random_str("gene_A", existing)
        existing[k1] = True
        k2 = CGEResults.generate_random_str("gene_A", existing)
        assert k1 != k2

    def test_add_phenotype_result(self):
        cge = CGEResults(args_dict={})
        ep = pd.Series({
            "Prediction Mean": 0.75,
            "Prediction STD": 0.05,
            "Prediction_0": 0.7,
            "Prediction_1": 0.8,
            "Binary Prediction Mean": 1,
            "Phenotype": "Human Pathogenic",
        })
        cge.add_phenotype_result(ep)
        assert "human_bacterial_pathogenicity" in cge.software_result["phenotypes_ml"]
        assert cge.software_result["phenotypes_ml"]["human_bacterial_pathogenicity"]["prediction"] == "Human Pathogenic"

    def test_add_bacterialneighbors_adds_entries(self):
        cge = CGEResults(args_dict={})
        import numpy as np
        neighbors_df = pd.DataFrame({
            "Names": ["Ecoli_K12", "Salmonella_LT2"],
            "RefSeq": ["NC_000913", "AE006468"],
            "Distances": [np.float64(0.12), np.float64(0.34)],
            "Taxonomy": [np.int64(562), np.int64(99287)],
            "Species": ["Escherichia coli", "Salmonella enterica"],
            "Strain": ["K12", "LT2"],
        })
        cge.add_bacterialneighbors("sample1", neighbors_df)
        assert len(cge.software_result["neighbors"]) == 2

    def test_add_proteinsatt_adds_seq_regions(self):
        cge = CGEResults(args_dict={})
        proteins_df = pd.DataFrame({
            "stitle": ["Protein A", "Protein B"],
            "pident": [98.0, 75.0],
            "length": [100, 80],
            "slen": [100, 90],
            "scovhsp": [100.0, 90.0],
            "sseqid": ["sp|A0A|GENE_EC", "sp|B0B|GENE_HU"],
            "sstart": [1, 1],
            "send": [100, 80],
            "qseqid": ["prot0", "prot1"],
            "qstart": [1, 1],
            "qend": [100, 80],
        })
        db = cge.add_database("swissprot", "2024_01")
        cge.add_proteinsatt(proteins_df, db)
        assert len(cge.software_result["seq_regions"]) == 2

    def test_add_gsearesults_adds_phenotypes(self):
        cge = CGEResults(args_dict={})
        gsea_df = pd.DataFrame({
            "Term": ["GO:0001", "GO:0002"],
            "Term_class": ["biological_process", "molecular_function"],
            "NES": [1.5, 2.0],
            "FDR q-val": [0.01, 0.05],
            "Lead_genes_description": ["GeneA OS=Ecoli;GeneB", "GeneC OS=Human;GeneD"],
        })
        db = cge.add_database("swissprot", "2024_01")
        cge.add_gsearesults(gsea_df, db)
        assert len(cge.software_result["phenotypes"]) == 2
