import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import logging
import json


from .preprocessdata.prott5_embedder import ProtT5_Embedder
from .preprocessdata.predict_proteins import Prodigal_EXEC
from .utils.os_utils import get_filename, create_outputfolder
from .utils.configuration_module import ConfigurationPF2
from .dl.model import Pathogen_DLModel
from .dl.utils.data_utils import NN_Data
from .dl.utils.report_utils import CGEResults
from .pathogenfinder2_mapping import PathogenFinder2_Mapping

DATA_FOLDER = "{}/../../data/".format(Path(__file__).parent.resolve())

def cl_arguments():
    parser = argparse.ArgumentParser(prog='Pathogenfinder2.0 Model',
                            description="Arguments for pathogenicity inference, training,"
                                        " testing and hyperparameter selection of the model.",
                            add_help=True)
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("-c", "--config", help="Json file with the configuration for PathogenFinder2 model.",
                                default=False)  # ToDO: Not require, add a standard json in case of default=None
    parent_parser.add_argument("-o", "--outputFolder", help="Folder where to output the results")

    parent_parser.add_argument("--cge", help="Output the cge format output", action="store_true")
    parent_parser.add_argument('-d', '--debug', help="For debugging",
                                action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parent_parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO)

    exec_paths = parent_parser.add_argument_group('Paths to executables',
                                    description="Paths to executables. They might not be necessary.")
    exec_paths.add_argument("--prodigalPath", help="Path to Prodigal", default="prodigal")
    exec_paths.add_argument("--protT5Path", help="Path to protT5", default="protT5")
    exec_paths.add_argument("--diamondPath", help="Patho to Diamond", default="diamond")

    subparsers = parser.add_subparsers(title="PathogenFinder functionalities", required=True)

    train_parser = subparsers.add_parser("train", help="Train the PathogenFinder2 model based on certain data and configuration.", parents=[parent_parser])
    train_parser.set_defaults(action="Train")

    test_parser = subparsers.add_parser("test", help="Test the PathogenFinder2 model.", parents=[parent_parser])
    test_parser.set_defaults(action="Test")

    inference_parser = subparsers.add_parser("inference", help="Predict using PathogenFinder2 model.", parents=[parent_parser])
    inference_parser.set_defaults(action="Inference")
    inference_parser.add_argument("-i", "--inputData", help="Input data for inference")
    inference_parser.add_argument("-w", "--weightsModel", help="Weights used by the deep learning model to predict")
    inference_parser.add_argument("-f", "--formatSeq", help="The format of the input data.",
                                    choices=["genome", "proteome", "embeddings"])
    inference_parser.add_argument("--multiFiles", help="If the input data are multiple files", action="store_true")    
    inference_parser.add_argument("--embeddings", help="If used, report or/and map the embeddings to the Bacterial Pathogenic Landscape",
                                  choices=[False, "report", "map"], default=False)
    inference_parser.add_argument("--attentions", help="If used, report the attentions or/and map the 20 proteins with highest attention score to UniRef50",
                                  choices=[False, "report", "map"], default=False)

    hyperopt_parser = subparsers.add_parser("hyperparam_opt",
                                    help="Hyperparam optimizaiton PathogenFinder2 model based on certain data and configuration.",
                                    parents=[parent_parser])
    hyperopt_parser.set_defaults(action="Hyperparam_Opt")

    return parser.parse_args()

class PathogenFinder2:

    def __init__(self, mode:str, config_data: tuple[str, dict]):

        self.pf2_config = ConfigurationPF2(mode=mode, user_config=config_data)
        if isinstance(config_data, str):
            self.pf2_config.load_json_params(json_file=user_config)
        else:
            self.pf2_config.load_dict_params(dict_args=config_data)

        self.model = Pathogen_DLModel(model_parameters=self.pf2_config.model_parameters,
                                      misc_parameters=self.pf2_config.misc_parameters,
                                      seed=self.pf2_config.model_parameters["Seed"])

    def predict_proteincontent(self, input_seq, log_folder, out_folder, prodigal_path="prodigal"):
        # Set up folder
        logging.info("Using device: {}".format("cpu"))

        prodigal_exec = Prodigal_EXEC(log_folder=log_folder, output_folder=out_folder,
                                prodigal_path=prodigal_path)
        protein_file = prodigal_exec(input_seq)
        return protein_file

    def inference_embeddings(self, embed_out, input_seq=None, input_txt=None, model_path=None,
                                pool_mode="mean", split_kmer=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info("Using device: {}".format(device))

        filename, ext = get_filename(input_seq)
        embedding_file = "{}/{}_protembeddings.h5".format(embed_out, filename)
        if input_seq is None and input_txt is None:
            raise argparse.ArgumentParser.error("Arguments input and file_paths cannot be used at the same time.")
        elif input_seq is not None and input_txt is not None:
            raise argparse.ArgumentParser.error("The argument input or the argument file_paths are needed.")
        elif input_seq is not None and input_txt is None:
            seq_path = Path(input_seq)
            embeder = ProtT5_Embedder()
            embeder.get_embeddings(seq_path=seq_path,  emb_path=embedding_file, pool_mode=pool_mode,
                                       split_kmer=split_kmer)
        else:
            file_path = Path(input_txt)
            embeder = ProtT5_Embedder()
            wrapper_multipleCore(embeder=embeder, path_file=file_path,
                        emb_path=embedding_file, pool_mode=pool_mode, split_kmer=split_kmer)
        return embedding_file

    def make_input(self, inference_parameters):
        if inference_parameters["Multiple Files"]:
            metadata = pd.read_csv(inference_parameters["Input Data"], sep="\t", names=["Input Files"])
        else:
            metadata = pd.DataFrame({"Input Files":[inference_parameters["Input Data"]]})
        metadata["File_Genome"] = None
        metadata["File_Proteins"] = None
        metadata["File_Embedding"] = None
        return metadata


    def inference(self, cge_output:bool=False):
        inference_parameters = self.pf2_config.inference_parameters

        input_metadata = self.make_input(inference_parameters=inference_parameters)
            
        for n in range(len(input_metadata)):
            message_pre = "================== Preprocessing {} ({}) file =====================".format(
                                                                                    n, os.path.basename(input_metadata.loc[n, "Input Files"]))
            print(message_pre)
            folder_sample = "{}/{}".format(self.pf2_config.misc_parameters["Results Folder"]["main"],
                                           os.path.basename(input_metadata.loc[n, "Input Files"]))
            os.mkdir(folder_sample) 
            preproc_folder = "{}/preprocessing".format(folder_sample)
            postproc_folder = "{}/postprocessing".format(folder_sample)
            log_folder = "{}/log".format(folder_sample)
            os.mkdir(preproc_folder)
            os.mkdir(postproc_folder)
            os.mkdir(log_folder)

            if inference_parameters["Sequence Format"] == "genome":
                print("Predicting the protein content (PRODIGAL).")
                input_metadata.loc[n, "File_Genome"] = input_metadata.loc[n,"Input Files"]
                input_metadata.loc[n, "File_Proteins"] = self.predict_proteincontent(
                                                                input_seq=input_metadata.loc[n, "File_Genome"], out_folder=preproc_folder,
                                                                log_folder=log_folder, prodigal_path=self.pf2_config.misc_parameters["Prodigal Path"])
            else:
                input_metadata.loc[n, "File_Genome"] = None
                input_metadata.loc[n, "File_Proteins"] = input_metadata.loc[n,"Input Files"]
            print(input_metadata.iloc[0]["File_Proteins"])
            if inference_parameters["Sequence Format"] == "genome" or inference_parameters["Sequence Format"] == "proteome":
                print("Predicting the embeddings of the protein content")
                try:
                    input_metadata.loc[n, "File_Embedding"] = self.inference_embeddings(
                                                    embed_out=preproc_folder, input_seq=input_metadata.loc[n, "File_Proteins"])
                except ZeroDivisionError:
                    raise TypeError("""Prodigal has not been able to predict any proteins in your file {}. """
                    """Please check that your file is an uncompressed FASTA file with a valid genomic sequence""".format(input_metadata.loc[n, "File_Genome"]))
            else:
                input_metadata.loc[n, "File_Proteins"] = None
                input_metadata.loc[n, "File_Embedding"] = input_metadata.loc[n, "Input Files"]
            message_pro = "".join(["="]*len(message_pre))
            print(message_pro)

        metadata_file = "{}/data_input.tsv".format(self.pf2_config.misc_parameters["Results Folder"]["conf"])
        input_metadata.to_csv(metadata_file, sep="\t", index=False)
        inference_parameters["Input Metadata"] = metadata_file
        print("Predicting the pathogenicity of the samples in the files {}".format(
                            ", ".join(input_metadata["Input Files"].apply(lambda x: os.path.basename(x)).tolist())))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info("Using device: {}".format(device))
        predicted_data = self.model.predict_model(inference_parameters=inference_parameters)

        if self.pf2_config.inference_parameters["Attentions"] == "map" or self.pf2_config.inference_parameters["Embeddings"] == "map":
            for n in range(len(input_metadata)):
                folder_sample = "{}/{}".format(self.pf2_config.misc_parameters["Results Folder"]["main"],
                                           os.path.basename(input_metadata.loc[n, "Input Files"]))
                postproc_folder = "{}/postprocessing".format(folder_sample)
                log_folder = "{}/log".format(folder_sample)

                if self.pf2_config.inference_parameters["Attentions"] == "map":
                    if self.pf2_config.inference_parameters["Sequence Format"] == "embeddings":
                        raise TypeError("There is no predicted protein sequence file available")
                    print("Aligning the 20 proteins with higher attention score from {} to UniRef50".format(
                                                            os.path.basename(input_metadata.iloc[0]["Input Files"])))
                    db_path = "{}/protein_db/uniref50".format(DATA_FOLDER)
                    mapped_data = PathogenFinder2_Mapping.map_proteins(diamond_path=self.pf2_config.misc_parameters["Diamond Path"],
                                                        folder_out="{}/out/".format(folder_sample), folder_tmp=postproc_folder,
                                                    db_path=db_path, prot_path=input_metadata.loc[n, "File_Proteins"],
                                                    att_path="{}/out/attentions.npz".format(folder_sample),
                                                    log_folder=log_folder, amount_prots=20, amount_hits=1)
                if self.pf2_config.inference_parameters["Embeddings"] == "map":
                    print("Mapping the sequence {} to the Bacterial Landscap Pathogen".format(
                                                                os.path.basename(input_metadata.iloc[0]["Input Files"])))
                    embed_file = "{}/out/embeddings.npz".format(folder_sample)
                    close_emb = PathogenFinder2_Mapping.map_embeddings(embeddings_bpl="{}/embeddings_BPL/embeddings.npz".format(DATA_FOLDER),
                                                embeddings_pred=embed_file, folder_out="{}/out".format(folder_sample))
    
        return predicted_data

    @staticmethod
    def create_cge_output(predicted_data:str, output_folder:str):
        for k, val in predicted_data:
            json_cge = CGEResults()
            json_cge.add_software_result()
            json_cge.add_phenotype_result(results_ensemble=val["Ensemble Predictions"])
            json_cge.save_result(output_path="{}/{}".format(output_folder, val["Features"]["Filename"]))


    def train(self, train_parameters):
        self.model.train_model(train_parameters=train_parameters)

    def test(self, test_parameters):
        test_parameters["Multiple Files"] = True
        predicted_data = self.inference(inference_parameters=test_parameters)
        self.model.test_model(predicted_data=predicted_data, test_parameters=test_parameters)

    def save_config(self):
        with open("{}/config.json".format(self.pf2_config.misc_parameters["Results Folder"]["conf"]), 'w') as f:
            json.dump(self.pf2_config, f)



def main():
    args = cl_arguments()
    logging.basicConfig(level=args.loglevel)
    ## Set Parameters ##

    if not args.config:
        config_data = vars(args)
    else:
        if os.path.isfile(args.config):
            config_data = args.config
        else:
            raise ValueError("If using --config, you need to point to an existing json file (pointed to {})".format(args.config))

    pathogenfinder_run = PathogenFinder2(mode=args.action, config_data=config_data)
    if args.action == "Train":
        pathogenfinder_run.train(train_parameters=pf2_config.train_parameters)
    elif args.action == "Test":
        pathogenfinder_run.test(test_parameters=pf2_config.test_parameters)
    elif args.action == "Hyperparam_opt":
        pathogenfinder_run.hyperparam_opt()
    elif args.action == "Inference":
        predicted_data = pathogenfinder_run.inference()
        if args.cge:
            PathogenFinder2.create_cge_output(predicted_data=predicted_data,
                                              output_folder=pathogefinder_run.pf2_config.misc_parameters["Results Folder"]["main"]) 
    else:
        raise ValueError("No valid option ({}) for using pathogenfinder was selected".format(args.action))

    pathogenfinder_run.save_config()

    print("Results saved on folder {}".format(pathogenfinder_run.pf2_config.misc_parameters["Results Folder"]["main"]))


if __name__ == '__main__':
    main()

