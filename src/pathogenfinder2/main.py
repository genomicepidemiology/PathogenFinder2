"""
PathogenFinder2 main orchestration module.

Provides the :class:`PathogenFinder2` class that drives the end-to-end
pipeline: protein prediction (Prodigal), ProtT5 embedding, deep-learning
ensemble inference, post-processing (KNN landscape mapping, protein alignment,
GSEA), and CGE JSON output.  The module-level :func:`main` entry point wires
the CLI arguments produced by :mod:`cli` to the appropriate pipeline
methods.
"""
import logging
import os
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

from pathogenfinder2.cli import pf2_arguments
from pathogenfinder2.utils.output import PredictionReport, CGEResults
from pathogenfinder2.exceptions import ExternalToolError, ConfigurationError




class PathogenFinder2:
    """End-to-end PathogenFinder2 pipeline.

    Orchestrates protein prediction (Prodigal), ProtT5 embedding, deep-learning
    ensemble inference, and optional post-processing steps (landscape mapping,
    protein alignment/GSEA).  Instantiate once per run, then call :meth:`predict`
    (or the individual step methods) to drive the pipeline.

    Parameters
    ----------
    mode:
        Pipeline mode, one of :attr:`MODES`.
    outPath:
        Directory under which all run outputs will be written.
    configuration_file:
        Path to a JSON user-config, a pre-parsed dict, or ``False`` to use the
        built-in default configuration.
    cge_output:
        If ``True``, accumulate CGE-format JSON results.
    """

    MODES = ["Align_Proteins", "Map_Embeddings", "Prediction", "Train", "Test", "Infer"]

    def __init__(self, mode: str, outPath: str,
                 configuration_file: str | dict | None = None,
                 cge_output: bool = False) -> None:

        from pathogenfinder2.utils.configuration import ConfigurationPF2, FilesModule

        if mode not in PathogenFinder2.MODES:
            raise ConfigurationError("The mode '{}' is not available as part of PathogenFinder2".format(mode))
        self.pf2_config = ConfigurationPF2(mode=mode, user_config=configuration_file)

        self.files_module = FilesModule(outputFolder=outPath, mode=mode)

        if cge_output:
            self.cge_results = {}
        else:
            self.cge_results = None


    def predict_protein_content(self, folder_key: str = "preprocess",
                                prodigal_path: str = "prodigal") -> dict[str, str]:
        """Run Prodigal on every input genome and return a per-sample status dict.

        Parameters
        ----------
        folder_key:
            Key in ``files_module["folders"]`` pointing to the preprocessing directory.
        prodigal_path:
            Path (or name on ``$PATH``) of the Prodigal executable.

        Returns
        -------
        dict[str, str]
            Maps each sample name to ``"Success"`` or an error message.
        """
        from pathogenfinder2.preprocessing.prodigal import ProdigalRunner

        start = time.time()
        logger.info("Predicting protein content with Prodigal")

        success_prediction = {}
        for baseseq in self.files_module["input_files"]:
            log_fold = self.files_module.log_folder(baseseq)
            preproc_fold = self.files_module["folders"][folder_key][baseseq]
            input_seq = self.files_module["data_files"]["genome_sequence"][baseseq]
            protein_file = self.files_module.proteome_file(baseseq)

            prodigal_exec = ProdigalRunner(log_folder=log_fold,
                                          output_folder=preproc_fold, prodigal_path=prodigal_path)
            (prot_file, gbk_file, stats_file,
                    outstd, errstd, command) = prodigal_exec(input_seq, prot_file=protein_file)
            amount_prots = ProdigalRunner.count_proteins(prot_file)

            elapsed = time.time() - start
            logger.debug("Prodigal finished for %s in %.1fs", os.path.basename(input_seq), elapsed)

            if amount_prots == 0:
                success_prediction[baseseq] = "ERROR: No proteins were predicted in the file {}".format(os.path.basename(input_seq))
            elif amount_prots > 14000:
                success_prediction[baseseq] = "ERROR: More than 14000 were predicted in the file {}, which is an amount beyond bacterial sizes.".format(os.path.basename(input_seq))
            else:
                success_prediction[baseseq] = "Success"

            if self.cge_results is None and success_prediction[baseseq] != "Success":
                raise ExternalToolError(success_prediction[baseseq])

        return success_prediction

    def infer_embeddings(self, success_proteins: dict | None = None,
                         model_path: str | None = None,
                         pool_mode: str = "mean", split_kmer: bool = True) -> None:
        """Compute ProtT5 embeddings for every (successful) sample."""
        from pathogenfinder2.preprocessing.embedder import ProtT5Embedder

        start = time.time()
        logger.info("Embedding proteins with ProtT5")

        for base_seq in self.files_module["input_files"]:
            if success_proteins is not None and success_proteins[base_seq] != "Success":
                continue
            embedding_file = self.files_module.embedding_file(base_seq)
            input_file = self.files_module.proteome_file(base_seq)
            embeder = ProtT5Embedder(model_dir=model_path)
            embeder.get_embeddings(seq_path=input_file, emb_path=embedding_file,
                                   pool_mode=pool_mode, split_kmer=split_kmer)

    def map_embeddings(self, embeddings_preds: str,
                       bpl_file: str | None = None,
                       success_proteins: dict | None = None) -> dict:
        """Project proteome embeddings onto the Bacterial Pathogenic Landscape."""
        logger.info("Mapping proteome to the Bacterial Pathogenic Landscape")
        from pathogenfinder2.postprocessing.landscape import MapEmbeddings

        if bpl_file is None:
            embeddings_bpl = self.pf2_config.get_bplfile()
        else:
            embeddings_bpl = bpl_file
        fitted_model = self.pf2_config.get_bpl_fitted_model()
        train_data = self.pf2_config.get_bpl_coordinates()
        closer_dfs = {}
        for base_seq in self.files_module["input_files"]:
            if success_proteins is not None and success_proteins[base_seq] != "Success":
                continue
            emb_pred = self.files_module["data_files"]["genome_embeddings"][base_seq]
            mapemb = MapEmbeddings(out_folder=self.files_module.results_folder(base_seq),
                                   data_embed=embeddings_bpl,
                                   fitted_model=fitted_model,
                                   train_data=train_data)
            test_transf = mapemb.fit_test_data(testdata=emb_pred)
            closer_df, closer_arr = mapemb.knn(test_transf)
            mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)
            closer_dfs[base_seq] = closer_df
            if self.cge_results is not None:
                self.cge_results[base_seq].add_bacterialneighbors(query_id=base_seq, neighbors_df=closer_df)
        return closer_dfs

    def align_proteins(self, db_path: str, diamond_path: str = "diamond",
                       amount_prots: int = 20, gsea: bool = False,
                       db_protmetadata: str | None = None, gsea_minsize: int = 15,
                       success_proteins: dict | None = None) -> dict:
        """Align top-attention proteins against a Diamond database."""
        logger.info("Aligning proteins of interest")
        from pathogenfinder2.postprocessing.alignment import MapProteins

        self.files_module.postprocess_output()

        mapped_datalst = {}
        for base_seq in self.files_module["input_files"]:
            if success_proteins is not None and success_proteins[base_seq] != "Success":
                continue
            att_path = self.files_module.attention_file(base_seq)
            prot_path = self.files_module.proteome_file(base_seq)
            folder_tmp = self.files_module.postprocess_folder(base_seq)

            mapprot = MapProteins(folder_out=self.files_module.results_folder(base_seq),
                                  folder_tmp=folder_tmp, diamond_path=diamond_path,
                                  db_path=db_path, metadata=db_protmetadata,
                                  log_folder=self.files_module.log_folder(base_seq))
            aln_results, gsea_results = mapprot.map_proteins(att_file=att_path,
                                            prot_file=prot_path, gsea=gsea, top_proteins=amount_prots,
                                            gsea_minsize=int(gsea_minsize))
            mapped_datalst[base_seq] = {"Alignment_Results": aln_results, "GSEA_Results": gsea_results}
            if self.cge_results is not None:
                database = self.cge_results[base_seq].add_database(name=db_path, version="")
                self.cge_results[base_seq].add_proteinsatt(proteins_df=aln_results, ref_db=database)
                if gsea:
                    self.cge_results[base_seq].add_gsearesults(gsea_df=gsea_results, ref_db=database)
        return mapped_datalst

    def run_neural_network(self, produce_embeddings: bool, produce_attentions: bool,
                           success_proteins: dict | None = None) -> dict:
        """Run the deep-learning ensemble and return per-sample predictions."""
        from pathogenfinder2.dl.model import PathogenDLModel

        logger.info("Inferring pathogenicity with neural networks")
        self.files_module.add_nn_products(produce_embeddings=produce_embeddings,
                                          produce_attentions=produce_attentions)
        model_dl = PathogenDLModel(model_parameters=self.pf2_config["Model Parameters"],
                                   misc_parameters=self.pf2_config["Misc Parameters"],
                                   seed=self.pf2_config["Model Parameters"]["Seed"])
        ensemble_results = model_dl.predict_model(input_metapath=self.files_module["input_metadata"],
                                                  save_embeddings=produce_embeddings,
                                                  save_attentions=produce_attentions)
        PredictionReport.get_predictions(ensemble_results=ensemble_results)
        if self.cge_results is not None:
            for base_seq in self.files_module["input_files"]:
                self.cge_results[base_seq].add_phenotype_result(
                    results_ensemble=ensemble_results[base_seq]["Ensemble Predictions"]
                )

        return ensemble_results

    def save_cge_results(self) -> None:
        """Write CGE JSON output for every sample."""
        if self.cge_results is None:
            return
        for k, val in self.cge_results.items():
            outpath = "{}/cge_output.json".format(self.files_module.results_folder(k))
            val.save_results(outpath)

    def _run_preprocessing(
        self,
        format_seq: str,
        prodigal_path: str,
        prott5_path: str,
    ) -> dict | None:
        """Run protein prediction and/or embedding generation.

        Returns a per-sample success dict when CGE output tracking is active,
        or ``None`` otherwise.
        """
        if format_seq == "genome":
            success_proteins = self.predict_protein_content(prodigal_path=prodigal_path)
            if self.cge_results is None:
                success_proteins = None
        else:
            success_proteins = {} if self.cge_results is not None else None

        if format_seq in ["genome", "proteome"]:
            self.infer_embeddings(model_path=prott5_path, success_proteins=success_proteins)

        return success_proteins

    def predict(self, input_file: str | bool, multi_file: str | bool = False,
                format_seq: str = "genome", prodigal_path: str | None = None,
                prott5_path: str | None = None, produce_embeddings: bool = False,
                produce_attentions: bool = False) -> tuple[dict, dict | None]:
        """Run the full prediction pipeline for one or more input files.

        Returns
        -------
        tuple[dict, dict | None]
            ``(ensemble_results, success_proteins)`` where *success_proteins* is
            ``None`` when CGE output is disabled.
        """
        if format_seq not in ["genome", "proteome", "embeddings"]:
            raise ValueError("The format_seq variable '{}' is not part of the available options".format(format_seq))

        self.files_module.create_nestedoutput(format_seq=format_seq, input_file=input_file,
                                              multi_file=multi_file)
        if isinstance(self.cge_results, dict):
            for basein in self.files_module["input_files"]:
                self.cge_results[basein] = CGEResults(args_dict=self.pf2_config)
                self.cge_results[basein].add_software_exec(parameters={
                    "inputfasta": self.files_module["data_files"]["input_sequence"][basein],
                    "inputfastq_1": None, "inputfastq_2": None,
                })

        success_proteins = self._run_preprocessing(format_seq, prodigal_path, prott5_path)
        input_df = self.files_module.save_inputmetadata(success_proteins=success_proteins)
        if len(input_df) > 0:
            ensemble_results = self.run_neural_network(produce_embeddings=produce_embeddings,
                                                       produce_attentions=produce_attentions)
        else:
            ensemble_results = {}

        if self.cge_results is not None:
            for base_seq in self.files_module["input_files"]:
                if success_proteins is not None:
                    log = success_proteins[base_seq]
                    if success_proteins[base_seq] != "Success":
                        summary = "Failed"
                    else:
                        summary = "Prediction: {}".format(ensemble_results[base_seq]["Ensemble Predictions"]["Phenotype"])
                else:
                    log = "Success"
                    summary = "Prediction: {}".format(ensemble_results[base_seq]["Ensemble Predictions"]["Phenotype"].item())

                self.cge_results[base_seq].add_log(summary, log=log)

        return ensemble_results, success_proteins

    def train(self) -> None:
        """Train the neural network using the loaded configuration."""
        from pathogenfinder2.dl.model import PathogenDLModel

        self.pf2_config["Misc Parameters"]["Results Folder"] = self.files_module["base_folder"]

        model_dl = PathogenDLModel(model_parameters=self.pf2_config["Model Parameters"],
                                   misc_parameters=self.pf2_config["Misc Parameters"],
                                   seed=self.pf2_config["Model Parameters"]["Seed"])
        model_dl.train_model(train_parameters=self.pf2_config["Train Parameters"])

    def infer(self, input_file: str | bool = False, multi_file: str | bool = False,
              prodigal_path: str | None = None, prott5_path: str | None = None) -> None:
        """Run protein prediction and embedding inference only (no classification)."""
        self.files_module.create_inferenceoutput(input_file=input_file, multi_file=multi_file)
        _ = self.predict_protein_content(prodigal_path=prodigal_path, folder_key="files")
        _ = self.infer_embeddings(model_path=prott5_path)

    @staticmethod
    def setup_swissprot(out_folder: str, tsv_path: str | None = None,
                        go_file: str | None = None, diamond_path: str | None = None) -> None:
        """Download and index a SwissProt protein database for alignment."""
        from pathogenfinder2.setup_prot_db import SetupSwissProt

        if tsv_path is None:
            if diamond_path is None:
                raise ValueError("Please provide a path to the diamond executable")
            fasta_path, tsv_path = SetupSwissProt.download_swissprot_bacteria(out_folder=out_folder)
            logger.info("SwissProt FASTA stored in %s", fasta_path)
            diamond_index = SetupSwissProt.diamond_index(fasta=fasta_path, db_name=str(Path(fasta_path).with_suffix("")),
                                                         diamond_path=diamond_path)
            logger.info("Diamond-indexed SwissProt DB stored in %s", str(Path(diamond_index).with_suffix("")))
        if go_file is None:
            go_file = SetupSwissProt.download_go_basic(out_folder=out_folder)
        tsv_formatted = "{}/uniprot_metadata_formated.tsv".format(out_folder)
        tsv_format = SetupSwissProt.format_metadata(tsv_path, go_file, output_path=tsv_formatted)
        logger.info("Formatted SwissProt TSV stored in %s", os.path.abspath(tsv_format))


def main() -> None:
    args = pf2_arguments()
    logging.basicConfig(level=args.loglevel,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.action == "Setup_SwissProt":
        PathogenFinder2.setup_swissprot(out_folder=args.outputFolder, tsv_path=args.swissprot_tsv,
                                        go_file=args.go_file, diamond_path=args.diamondPath)
    else:
        pathogenfinder2_main = PathogenFinder2(mode=args.action, outPath=args.outputFolder,
                                               configuration_file=args.config, cge_output=args.cge)
        if args.action == "Prediction":
            predictions, success_proteins = pathogenfinder2_main.predict(
                input_file=args.inputFile, multi_file=args.multipleFiles,
                format_seq=args.formatSeq, prodigal_path=args.prodigalPath,
                prott5_path=args.protT5Path, produce_embeddings=args.embedProteome,
                produce_attentions=args.attProteins)

            results_module = PredictionReport(out_folder=pathogenfinder2_main.files_module["folders"]["results"])
            predictions_paths, embeddings_paths, att_paths = results_module.save_report(
                results_ensemble=predictions,
                save_attentions=args.attProteins,
                save_embeddings=args.embedProteome)
            if args.embedProteome == "map":
                _ = pathogenfinder2_main.map_embeddings(embeddings_preds=embeddings_paths,
                                                        success_proteins=success_proteins)
            if args.attProteins == "align":
                _ = pathogenfinder2_main.align_proteins(db_path=args.dbProteins,
                                                                diamond_path=args.diamondPath,
                                                                gsea=args.gsea, db_protmetadata=args.dbMetadataProteins,
                                                                gsea_minsize=args.minsize_gsea,
                                                                success_proteins=success_proteins)
            if args.cge:
                pathogenfinder2_main.save_cge_results()
        elif args.action == "Train":
            pathogenfinder2_main.train()
        elif args.action == "Infer":
            pathogenfinder2_main.infer(input_file=args.inputFile, multi_file=args.multipleFiles,
                                       prodigal_path=args.prodigalPath, prott5_path=args.protT5Path)
        else:
            raise ConfigurationError("The mode '{}' is not available as part of PathogenFinder2".format(args.action))


if __name__ == '__main__':
    main()
