import logging
import os
import sys
import time
from pathlib import Path

from pathogenfinder2.pf2_arguments import pf2_arguments
from pathogenfinder2.utils.os_utils import read_multifiles, center_print
from pathogenfinder2.utils.output_module import Prediction_Report, CGEResults




class PathogenFinder2_Main:

    MODES = ["Align_Proteins", "Map_Embeddings", "Prediction", "Train", "Test", "Infer"]

    def __init__(self, mode:str, outPath:str, configuration_file:[str, dict, bool]=False,
                        cge_output:bool=False) -> None:

        from pathogenfinder2.utils.configuration import ConfigurationPF2, Files_Module

        if mode not in PathogenFinder2_Main.MODES:
            raise ValueError("The mode '{}' is not available as part of PathogeFinder2".format(mode))
        self.pf2_config = ConfigurationPF2(mode=mode, user_config=configuration_file)
    
        self.files_module = Files_Module(outputFolder=outPath, mode=mode)

        if cge_output:
            self.cge_results = {}
        else:
            self.cge_results = False
        
        
    def predict_proteincontent(self, folder_key="preprocess", prodigal_path="prodigal"):
        from pathogenfinder2.preprocessdata.predict_proteins import Prodigal_Executable

        start = time.time()
        print(center_print("### Predicting protein content with Prodigal ###"), end="", flush=True)

        success_prediction = {}
        for baseseq in self.files_module["input_files"]:
            log_fold = self.files_module["folders"]["log"][baseseq]
            preproc_fold = self.files_module["folders"][folder_key][baseseq]
            input_seq = self.files_module["data_files"]["genome_sequence"][baseseq]
            protein_file = self.files_module["data_files"]["proteome_sequence"][baseseq]

            prodigal_exec = Prodigal_Executable(log_folder=log_fold,
                                                output_folder=preproc_fold, prodigal_path=prodigal_path)
            (prot_file, gbk_file, stats_file,
                    outstd, errstd, command) = prodigal_exec(input_seq, prot_file=protein_file)
            amount_prots = Prodigal_Executable.count_proteins(prot_file)

            # ---- update ONLY the seconds counter ----
            elapsed = time.time() - start
            print("\r" + center_print(f"### Predicting protein content with Prodigal ({elapsed:.1f}s) ###"),
                            end="", flush=True)

            if amount_prots == 0:
                success_prediction[baseseq] = "ERROR: No proteins were predicted in the file {}".format(os.path.basename(input_seq))
            elif amount_prots > 14000:
                success_prediction[baseseq] = "ERROR: More than 14000 were predicted in the file {}, which is an amount beyond bacterial sizes.".format(os.path.basename(input_seq))
            else:
                success_prediction[baseseq] = "Success"
            
            if not self.cge_results and success_prediction[baseseq] != "Success":
                sys.exit(success_prediction[baseseq]) 

            #if self.cge_results:
             #   self.cge_results[baseseq].add_software_exec(software_name="Prodigal", command=command,
              #                              stdout=outstd, stderr=errstd, parameters={"inputfile": input_seq})

        return success_prediction
            
    def infere_embeddings(self, success_proteins:dict=False, model_path:str=None,
                                    pool_mode:str="mean", split_kmer:bool=True):
        
        from pathogenfinder2.preprocessdata.prott5_embedder import ProtT5_Embedder

        start = time.time()
        print(center_print("### Embedding Proteins with protT5 ###"))

        for base_seq in self.files_module["input_files"]:
            if success_proteins and success_proteins[base_seq] != "Success":
                continue
            embedding_file = self.files_module["data_files"]["embedding_file"][base_seq]
            input_file = self.files_module["data_files"]["proteome_sequence"][base_seq]
            embeder = ProtT5_Embedder(model_dir=model_path)
            embeder.get_embeddings(seq_path=input_file,  emb_path=embedding_file, pool_mode=pool_mode,
                                       split_kmer=split_kmer)
           # if self.cge_results:
            #    self.cge_results[base_seq].add_software_exec(software_name="protT5", command="",
             #                               stdout="", stderr="", parameters={})

    def map_embeddings(self, embeddings_preds:str, bpl_file:str=None, success_proteins:[dict, bool]=False):
        print(center_print("### Map proteome to the Bacterial Pathogenic Landscape ###"))
        from pathogenfinder2.postprocessdata.embedding_PF2feature import MapEmbeddings

        if bpl_file is None:
            embeddings_bpl = self.pf2_config.get_bplfile()
        else:
            embeddings_bpl = bpl_file
        closer_dfs = {}
        for base_seq in self.files_module["input_files"]:
            if success_proteins and success_proteins[base_seq] != "Success":
                continue
            emb_pred = self.files_module["data_files"]["genome_embeddings"][base_seq]
            mapemb = MapEmbeddings(out_folder=self.files_module["folders"]["results"][base_seq],
                                    data_embed=embeddings_bpl)
            test_transf = mapemb.fittestdata(testdata=emb_pred)
            closer_df, closer_arr = mapemb.knn(test_transf)
            mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)
            closer_dfs[base_seq] = closer_df
            if self.cge_results:
                self.cge_results[base_seq].add_bacterialneighbors(query_id=base_seq, neighbors_df=closer_df)
        return closer_dfs
    
    def map_attentions(self, db_path:str, diamond_path:str="diamond",
                        amount_prots:int=20, gsea:bool=False,
                        db_protmetadata:str=None, gsea_minsize:int=15,
                        success_proteins:[dict, bool]=False):
        print(center_print("### Align proteins of interest ###"))
        from pathogenfinder2.postprocessdata.protein_PF2feature import MapProteins

        self.files_module.postprocess_output()

        mapped_datalst = {}
        for base_seq in self.files_module["input_files"]:
            if success_proteins and success_proteins[base_seq] != "Success":
                continue
            att_path = self.files_module["data_files"]["attention_file"][base_seq]
            prot_path = self.files_module["data_files"]["proteome_sequence"][base_seq]
            folder_tmp = self.files_module["folders"]["postprocess"][base_seq]

            mapprot = MapProteins(folder_out=self.files_module["folders"]["results"][base_seq],
                                    folder_tmp=folder_tmp, diamond_path=diamond_path,
                                    db_path=db_path, metadata=db_protmetadata,
                                    log_folder=self.files_module["folders"]["log"][base_seq])
            aln_results, gsea_results = mapprot.map_proteins(att_file=att_path,
                                            prot_file=prot_path, gsea=gsea, top_proteins=amount_prots,
                                            gsea_minsize=int(gsea_minsize))
            mapped_datalst[base_seq] = {"Alignment_Results": aln_results, "GSEA_Results": gsea_results}
            if self.cge_results:
                #self.cge_results[base_seq].add_software_exec(software_name="Diamond", command="",
                 #                       stdout="", stderr="", parameters={})
                #TODO version uniref50
                database = self.cge_results[base_seq].add_database(name=db_path,
                                                                    version="")
                self.cge_results[base_seq].add_proteinsatt(proteins_df=aln_results,
                                                            ref_db=database)
                if gsea:
                    #self.cge_results[base_seq].add_software_exec(software_name="GSEA", command="",
                     #                   stdout="", stderr="", parameters={})
                    self.cge_results[base_seq].add_gsearesults(gsea_df=gsea_results, ref_db=database)
        return mapped_datalst

    def dl_predict(self, produce_embeddings, produce_attentions, success_proteins:[dict, bool]=False):

        from pathogenfinder2.dl.model import Pathogen_DLModel

        print(center_print("### Inferring pathogenicity with Neural Networks ###"))
        self.files_module.add_nn_products(produce_embeddings=produce_embeddings,
                                            produce_attentions=produce_attentions)
        model_dl = Pathogen_DLModel(model_parameters=self.pf2_config["Model Parameters"],
                                      misc_parameters=self.pf2_config["Misc Parameters"],
                                      seed=self.pf2_config["Model Parameters"]["Seed"])
        predicted_data = model_dl.predict_model(input_metapath=self.files_module["input_metadata"],
                                                save_embeddings=produce_embeddings,
                                                save_attentions=produce_attentions)
        ensemble_results = Prediction_Report.get_predictions(ensemble_results=predicted_data)
        if self.cge_results:
            for base_seq in self.files_module["input_files"]:
                self.cge_results[base_seq].add_phenotype_result(results_ensemble=ensemble_results[base_seq]["Ensemble Predictions"])


        return ensemble_results
    
    def save_cge_results(self):
        for k, val in self.cge_results.items():
            outpath = "{}/cge_output.json".format(self.files_module["folders"]["results"][k])
            val.save_results(outpath)

    def predict(self, input_file:[bool,str], multi_file:[bool, str]=False, format_seq:str="genome",
                    prodigal_path:str=None, prott5_path:str=None, produce_embeddings:bool=False,
                    produce_attentions:bool=False):


        self.files_module.add_input(input_file=input_file, multi_file=multi_file)

        if format_seq not in ["genome", "proteome", "embeddings"]:
            raise ValueError("The format_seq variable '{}' is not part of the available options".format(format_seq))

        self.files_module.create_nestedoutput(format_seq=format_seq, input_file=input_file,
                                                multi_file=multi_file)
        if isinstance(self.cge_results, dict):
            for basein in self.files_module["input_files"]:
                self.cge_results[basein] = CGEResults(args_dict=self.pf2_config)
                self.cge_results[basein].add_software_exec(parameters={
                                "inputfasta":self.files_module["data_files"]["input_sequence"][basein],
                                "inputfastq_1":None, "inputfastq_2":None})
        else:
            pass

        if format_seq == "genome":
            success_proteins = self.predict_proteincontent(prodigal_path=prodigal_path)
            if self.cge_results:
                success_proteins = success_proteins
            else:
                success_proteins = False
        else:
            if self.cge_results:
                success_proteins = {}
            else:
                success_proteins = False        
        if format_seq in ["genome", "proteome"]:
            embed_file = self.infere_embeddings(model_path=prott5_path, success_proteins=success_proteins)
        else:
            pass
        input_df = self.files_module.save_inputmetadata(success_proteins=success_proteins)
        if len(input_df) > 0:
            ensemble_results = self.dl_predict(produce_embeddings=produce_embeddings,
                                                produce_attentions=produce_attentions)
        else:
            ensemble_results = {}

        if self.cge_results:
            for base_seq in self.files_module["input_files"]:
                if success_proteins:
                    log = success_proteins[base_seq]
                    if success_proteins[base_seq] != "Success":
                        summary = "Failed"
                    else:
                        summary = "Prediciton: {}".format(ensemble_results[base_seq]["Ensemble Predictions"]["Phenotype"])
                else:
                    summary = "Prediciton: {}".format(ensemble_results[base_seq]["Ensemble Predictions"]["Phenotype"].item())

                self.cge_results[base_seq].add_log(summary, log=success_proteins[base_seq])
                
        return ensemble_results, success_proteins
    
    def train(self):
        self.pf2_config["Misc Parameters"]["Results Folder"] = self.files_module["base_folder"]

        model_dl = Pathogen_DLModel(model_parameters=self.pf2_config["Model Parameters"],
                                      misc_parameters=self.pf2_config["Misc Parameters"],
                                      seed=self.pf2_config["Model Parameters"]["Seed"])
        model_dl.train_model(train_parameters=self.pf2_config["Train Parameters"])

    def infere(self, input_file=False,  multi_file=False, prodigal_path:str=None, prott5_path:str=None):
        
        self.files_module.add_input(input_file=input_file, multi_file=multi_file)
        self.files_module.create_inferenceoutput(input_file=input_file, multi_file=multi_file)
        _ = self.predict_proteincontent(prodigal_path=prodigal_path, folder_key="files")
        embed_file = self.infere_embeddings(model_path=prott5_path)

    @staticmethod
    def setup_swissprot(out_folder, tsv_path:str=False, go_file:str=False, diamond_path:str=None):
        from pathogenfinder2.setup_prot_db import SetupSwissProt

        if not tsv_path:
            if diamond_path is None:
                raise ValueError("Please provide a path to the diamond executable")
            fasta_path, tsv_path = SetupSwissProt.download_swissprot_bacteria(out_folder=out_folder)
            print(center_print("####  SWISSPROT FASTA IS STORED IN {}  ####".format(fasta_path)))
            diamond_index = SetupSwissProt.diamond_index(fasta=fasta_path, db_name=str(Path(fasta_path).with_suffix("")),
                                                            diamond_path=diamond_path)
            print(center_print("####  DIAMOND INDEXED SWISSPROT FASTA IS STORED IN {}  ####".format(
                                str(Path(diamond_index).with_suffix("")))))
        if not go_file:
            go_file = SetupSwissProt.download_go_basic(out_folder=out_folder)
        tsv_formatted = "{}/uniprot_metadata_formated.tsv".format(out_folder)
        tsv_format = SetupSwissProt.format_metadata(tsv_path, go_file,
                                        output_path=tsv_formatted)
        print(center_print("####  FORMATED SWISSPROT TSV IS STORED IN {}  ####".format(os.path.abspath(tsv_formatted))))

def main():
    args = pf2_arguments()
    logging.basicConfig(level=args.loglevel)

    if args.action == "Align_Proteins":
        pass
    elif args.action == "Map_Embeddings":
        pass
    elif args.action == "Setup_SwissProt":
        PathogenFinder2_Main.setup_swissprot(out_folder=args.outputFolder, tsv_path=args.swissprot_tsv,
                                                go_file=args.go_file, diamond_path=args.diamondPath)
    else:
        pathogenfinder2_main = PathogenFinder2_Main(mode=args.action, outPath=args.outputFolder,
                                                    configuration_file=args.config, cge_output=args.cge)
        if args.action == "Prediction":
            predictions, success_proteins = pathogenfinder2_main.predict(
                                                    input_file=args.inputFile,  multi_file=args.multipleFiles, 
                                                    format_seq=args.formatSeq, prodigal_path=args.prodigalPath,
                                                    prott5_path=args.protT5Path, produce_embeddings=args.embedProteome,
                                                    produce_attentions=args.attProteins)

            results_module = Prediction_Report(out_folder=pathogenfinder2_main.files_module["folders"]["results"])
            predictions_paths, embeddings_paths, att_paths = results_module.save_report(
                                                                results_ensemble=predictions,
                                                                save_attentions=args.attProteins,
                                                                save_embeddings=args.embedProteome)
            if args.embedProteome == "map":
                embed_data = pathogenfinder2_main.map_embeddings(embeddings_preds=embeddings_paths,
                                                                success_proteins=success_proteins)
            if args.attProteins == "align":
                prot_data = pathogenfinder2_main.map_attentions(db_path=args.dbProteins,
                                                                diamond_path=args.diamondPath,
                                                                gsea=args.gsea, db_protmetadata=args.dbMetadataProteins,
                                                                gsea_minsize=args.minsize_gsea,
                                                                success_proteins=success_proteins)
            if args.cge:
                pathogenfinder2_main.save_cge_results()
        elif args.action == "Train":
            pathogenfinder2_main.train()
        elif args.action == "Test":
            pass
        elif args.action == "Infer":
            pathogenfinder2_main.infere(input_file=args.inputFile,  multi_file=args.multipleFiles,
                                        prodigal_path=args.prodigalPath, prott5_path=args.protT5Path)
        else:
            raise ValueError("The mode '{}' is not available as part of PathogeFinder2".format(args.action))


if __name__ == '__main__':
    main()
