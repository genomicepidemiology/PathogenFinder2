import logging
import os

from pathogenfinder2.pf2_arguments import pf2_arguments
from pathogenfinder2.utils.output_module import Prediction_Report, CGEResults
from pathogenfinder2.utils.os_utils import read_multifiles
from pathogenfinder2.utils.configuration import ConfigurationPF2, Files_Module

from pathogenfinder2.preprocessdata.predict_proteins import Prodigal_Executable
from pathogenfinder2.preprocessdata.prott5_embedder import ProtT5_Embedder
from pathogenfinder2.dl.model import Pathogen_DLModel



class PathogenFinder2_Main:

    MODES = ["Align_Proteins", "Map_Embeddings", "Prediction", "Train", "Test", "Infere"]

    def __init__(self, mode:str, outPath:str, configuration_file:[str, dict, bool]=False) -> None:
        if mode not in PathogenFinder2_Main.MODES:
            raise ValueError("The mode '{}' is not available as part of PathogeFinder2".format(mode))
        self.pf2_config = ConfigurationPF2(mode=mode, user_config=configuration_file)

        if mode in ["Prediction", "Train", "Test"]:
            self.model_dl = Pathogen_DLModel(model_parameters=self.pf2_config["Model Parameters"],
                                      misc_parameters=self.pf2_config["Misc Parameters"],
                                      seed=self.pf2_config["Model Parameters"]["Seed"])
        else:
            self.model_dl = False
        
        self.files_module = Files_Module(outputFolder=outPath, mode=mode)
        
        
    def predict_proteincontent(self, prodigal_path="prodigal"):
        for baseseq in self.files_module["input_files"]:
            log_fold = self.files_module["folders"]["log"][baseseq]
            preproc_fold = self.files_module["folders"]["preprocess"][baseseq]
            input_seq = self.files_module["data_files"]["genome_sequence"][baseseq]
            protein_file = self.files_module["data_files"]["proteome_sequence"][baseseq]

            prodigal_exec = Prodigal_Executable(log_folder=log_fold,
                                                output_folder=preproc_fold, prodigal_path=prodigal_path)
            (prot_file, gbk_file, 
                    stats_file, outstd, errstd) = prodigal_exec(input_seq, prot_file=protein_file)
            
    def infere_embeddings(self, model_path:str=None, pool_mode:str="mean", split_kmer:bool=True):
        # TODO Path to protrans
        for base_seq in self.files_module["input_files"]:
            embedding_file = self.files_module["data_files"]["embedding_file"][base_seq]
            input_file = self.files_module["data_files"]["proteome_sequence"][base_seq]
            embeder = ProtT5_Embedder()
            embeder.get_embeddings(seq_path=input_file,  emb_path=embedding_file, pool_mode=pool_mode,
                                       split_kmer=split_kmer)

    def map_embeddings(self, embeddings_preds:str, bpl_file:str=None):
        from pathogenfinder2.postprocessdata.embedding_PF2feature import MapEmbeddings

        if bpl_file is None:
            embeddings_bpl = ConfigurationPF2.get_bplfile()
        else:
            embeddings_bpl = bpl_file
        closer_dfs = []
        for emb_pred in embeddings_preds:
            mapemb = MapEmbeddings(out_folder=self.files_module["results_folder"], data_embed=embeddings_bpl)
            test_transf = mapemb.fittestdata(testdata=emb_pred)
            closer_df, closer_arr = mapemb.knn(test_transf)
            mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)
            closer_dfs.append(closer_df)
        return closer_dfs
    
    def map_attentions(self, db_path:str, diamond_path:str="diamond",
                        amount_hits:int=1, amount_prots:int=20):

        from pathogenfinder2.postprocessdata.protein_PF2feature import MapProteins

        self.files_module.postprocess_output()

        mapped_datalst = []
        for att_path, prot_path in zip(att_paths, prot_paths):

            mapprot = MapProteins(folder_out=self.files_module["results_folder"], folder_tmp=folder_tmp,
                                    diamond_path=diamond_path, db_path=db_path)
            tsv_file, fsa_file = mapprot.read_attentionfile(att_file=att_path,
                                                            prot_file=prot_path)
            diamond_file = mapprot.run_diamond(infile=fsa_file, num_report=amount_hits,
                                                log_folder=self.files_module["log_folder"])
            mapped_data = mapprot.analyze_results(infile=diamond_file, df_att=tsv_file)
            mapped_datalst.append(mapped_data)
        return mapped_datalst

    def predict(self, input_file:[bool,str], multi_file:[bool, str]=False, format_seq:str="genome", prodigal_path:str=None,
                    produce_embeddings:bool=False, produce_attentions:bool=False):


        self.files_module.add_input(input_file=input_file, multi_file=multi_file)

        if format_seq not in ["genome", "proteome", "embeddings"]:
            raise ValueError("The format_seq variable '{}' is not part of the available options".format(format_seq))

        self.files_module.create_nestedoutput(format_seq=format_seq, input_file=input_file,
                                                multi_file=multi_file)

        if format_seq == "genome":
            prot_file = self.predict_proteincontent(prodigal_path=prodigal_path)
        else:
            pass
        
        if format_seq in ["genome", "proteome"]:
            embed_file = self.infere_embeddings()
        else:
            pass
        
        self.files_module.save_inputmetadata()
        predicted_data = self.model_dl.predict_model(
                                input_metapath=self.files_module["input_metadata"],
                                save_embeddings=produce_embeddings,
                                save_attentions=produce_attentions)
        ensemble_results = Prediction_Report.get_predictions(ensemble_results=predicted_data)
        return ensemble_results
    
        
    def test(self):
        pass
    def train(self):
        pass
    def infere(self):
        pass


def main():
    args = pf2_arguments()
    logging.basicConfig(level=args.loglevel)
    if args.action == "Align_Proteins":
        pass
    elif args.action == "Map_Embeddings":
        pass
    else:
        pathogenfinder2_main = PathogenFinder2_Main(mode=args.action,
                                                    outPath=args.outputFolder)
        if args.action == "Prediction":
            predictions = pathogenfinder2_main.predict(input_file=args.inputFile,  multi_file=args.multipleFiles, 
                                                    format_seq=args.formatSeq, prodigal_path=args.prodigalPath,
                                                    produce_embeddings=args.embedProteome, produce_attentions=args.attProteins)
            results_module = Prediction_Report(out_folder=pathogenfinder2_main.files_module["sequences"])
            predictions_paths, embeddings_paths, att_paths = results_module.save_report(
                                                                results_ensemble=predictions,
                                                                save_attentions=args.attProteins,
                                                                save_embeddings=args.embedProteome)
            if args.embedProteome == "map":
                embed_data = pathogenfinder2_main.map_embeddings(embeddings_preds=embeddings_paths)
            if args.attProteins == "align":
                prot_data = pathogenfinder2_main.map_attentions(att_paths=att_paths, db_path=args.dbProteins,
                                                                diamond_path=args.diamondPath)
            if args.cge:
                pathogenfinder2_main.create_cge_output(predicted_data=predictions,
                                                    extra_phenotype=extra_phenotype,
                                                    output_folder=pathogenfinder_run.pf2_config.misc_parameters["Results Folder"]["main"]) 
    
        elif args.action == "Train":
            pass
        elif args.action == "Test":
            pass
        elif args.action == "Infere":
            pass
        else:
            raise ValueError("The mode '{}' is not available as part of PathogeFinder2".format(args.action))


if __name__ == '__main__':
    main()