import argparse
import os
from pathlib import Path
import pandas as pd
import torch


from preprocessdata.prott5_embedder import ProtT5_Embedder
from preprocessdata.extract_proteins import Prodigal_EXEC
from utils.file_utils import get_filename
from utils.configuration_module import ConfigurationPF2
from dl.dl_functions.model import Pathogen_DLModel
from dl.utils.data_utils import NN_Data



def cl_arguments():
    parser = argparse.ArgumentParser(prog='Pathogenfinder2.0 Model',
                            description="Arguments for pathogenicity inference, training,"
                                        " testing and hyperparameter selection of the model.",
                            add_help=True)
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("-c", "--config", help="Json file with the configuration for PathogenFinder2 model.",
                                required=True)  # ToDO: Not require, add a standard json in case of default=None
    parent_parser.add_argument("-o", "--outputFolder", help="Folder where to output the results", default=None)
    exec_paths = parent_parser.add_argument_group('Paths to executables',
                                    description="Paths to executables. They might not be necessary.")
    exec_paths.add_argument("--prodigalPath", help="Path to Prodigal", default="prodigal")
    exec_paths.add_argument("--protT5Path", help="Path to protT5", default="protT5")

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


    hyperopt_parser = subparsers.add_parser("hyperparam_opt",
                                    help="Hyperparam optimizaiton PathogenFinder2 model based on certain data and configuration.",
                                    parents=[parent_parser])
    hyperopt_parser.set_defaults(action="Hyperparam_Opt")
    return parser.parse_args()

class PathogenFinder2:

    def __init__(self, model_parameters, misc_parameters):
        self.output_folder = os.path.abspath(misc_parameters["Results Folder"])
        os.mkdir(self.output_folder)

        self.model = Pathogen_DLModel(model_parameters=model_parameters, misc_parameters=misc_parameters,
                                        seed=model_parameters["Seed"])
        self.model_parameters = model_parameters


    def predict_proteincontent(self, input_seq, preprocess_folder, prodigal_path="pyrodigal"):
        # Set up folder
        print("Using device: {}".format("cpu"))
        log_folder = "{}/prodigal_log_files".format(preprocess_folder)
        os.mkdir(log_folder)

        prodigal_exec = Prodigal_EXEC(log_folder=log_folder, output_folder=preprocess_folder,
                                prodigal_path=prodigal_path)
        protein_file = prodigal_exec(input_seq)
        return protein_file

    def inference_embeddings(self, embed_out, input_seq=None, input_txt=None, model_path=None,
                                pool_mode="mean", split_kmer=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Using device: {}".format(device))

        filename, ext = get_filename(input_seq)
        embedding_file = "{}/{}.h5".format(embed_out, filename)
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


    def inference(self, inference_parameters):
        input_metadata = self.make_input(inference_parameters=inference_parameters)
        preprocess_folder = "{}/preprocessdata".format(self.output_folder)
        os.mkdir(preprocess_folder)
        
        for n in range(len(input_metadata)):
            print("================== Infering {} file =====================".format(input_metadata.loc[n, "Input Files"]))
            input_metadata[n, "File_Genome"] = os.path.abspath(input_metadata.loc[n, "Input Files"])        

            if inference_parameters["Sequence Format"] == "genome":
                print("Predicting the protein content.")
                input_metadata.loc[n, "File_Proteins"] = self.predict_proteincontent(
                                                    input_seq=input_metadata.loc[n, "File_Genome"], preprocess_folder=preprocess_folder)
            else:
                input_metadata.loc[n, "File_Genome"] = None
                input_metadata.loc[n, "File_Proteins"] = input_metadata.loc[n,"Input Files"]

            if inference_parameters["Sequence Format"] == "genome" or inference_parameters["Sequence Format"] == "proteome":
                print("Predicting the embeddings of the protein content")
                input_metadata.loc[n, "File_Embedding"] = self.inference_embeddings(
                                                    embed_out=preprocess_folder, input_seq=input_metadata.loc[n, "File_Proteins"])
            else:
                input_metadata.loc[n, "File_Proteins"] = None
                input_metadata.loc[n, "File_Embedding"] = input_metadata.loc[n, "Input Files"]
        metadata_file = "{}/data_input.tsv".format(self.output_folder)

        input_metadata.to_csv(metadata_file, sep="\t", index=False)
        inference_parameters["Input Metadata"] = metadata_file
        print("Predicting the pathogenicity of the samples in the files {}".format(", ".join(input_metadata["Input Files"].tolist())))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Using device: {}".format(device))
        predicted_data = self.model.predict_model(inference_parameters=inference_parameters)
        return predicted_data

    def train(self, train_parameters):
        self.model.train_model(train_parameters=train_parameters)

    def test(self, test_parameters):
        test_parameters["Multiple Files"] = True
        predicted_data = self.inference(inference_parameters=test_parameters)
        self.model.test_model(predicted_data=predicted_data, test_parameters=test_parameters)






def main(args):
    ## Set Parameters ##
    pf2_config = ConfigurationPF2(mode=args.action)
    pf2_config.load_json_params(json_file=args.config)
    pf2_config.load_args_params(args=args)

    pathogenfinder = PathogenFinder2(model_parameters=pf2_config.model_parameters,
                                    misc_parameters=pf2_config.misc_parameters)
    if args.action == "Train":
        pathogenfinder.train(train_parameters=pf2_config.train_parameters)
    elif args.action == "Test":
        pathogenfinder.test(test_parameters=pf2_config.test_parameters)
    elif args.action == "Hyperparam_opt":
        pathogenfider.hyperparam_opt()
    elif args.action == "Inference":
        pathogenfinder.inference(inference_parameters=pf2_config.inference_parameters)
    else:
        raise ValueError("No valid option for using pathogenfinder was selected")


if __name__ == '__main__':
    args = cl_arguments()
    main(args)

