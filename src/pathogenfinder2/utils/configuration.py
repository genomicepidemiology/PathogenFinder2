import os
import json
import pandas as pd
from torch import nn, optim

from pathlib import Path
from collections import UserDict

from pathogenfinder2.utils.os_utils import create_outputfolder, read_multifiles
from pathogenfinder2.utils.configuration_utils import NNEncoder, ParamsModel



class ConfigurationPF2(UserDict):

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, mode:str, user_config:[str,dict,bool]=False):
        
        self.data = {}
        self.mode = mode
        if not user_config:
            config_data = ConfigurationPF2.load_baseconfig()
            config_data = ConfigurationPF2.load_baseweights(config_data=config_data)
        else:
            config_data = self.load_user_config(user_config=user_config)
        config_data["Misc Parameters"]["Actions"] = mode
        self.update(config_data)

    @staticmethod
    def load_baseconfig():
        config_file = Path(ConfigurationPF2.CURRENT_DIR) / '../data/configs/config_base.json'
        with open(config_file) as f:
            config_dict = json.load(f)
        return config_dict
    
    @staticmethod
    def load_baseweights(config_data):
        for i in [1, 2, 3, 4]:
            weight_file = Path(ConfigurationPF2.CURRENT_DIR) / '../data/models_weights/weights_model{}.pickle'.format(i)
            config_data["Model Parameters"]["Network Weights"].append(weight_file)
        return config_data

    def get_bplfile(self):
        return Path(ConfigurationPF2.CURRENT_DIR) / '../data/bpl/embeddings.npz'


    def load_user_config(self, user_config):
        if isinstance(user_config, str):
            with open(user_config) as f:
                config_dict = json.load(f)
        else:
            config_dict = user_config
        config_dict = ConfigurationPF2.add_torch_functions(config_dict)
        return config_dict

    @staticmethod
    def add_torch_functions(config_dict):
        if isinstance(config_dict["Train Parameters"]["Optimizer"], str):
            if config_dict["Train Parameters"]["Optimizer"] == "NAdam":
                config_dict["Train Parameters"]["Optimizer"] = optim.NAdam
            else:
                raise ValueError("The parameter {} for the Loss Function is not defined".format(
                                    config_dict["Train Parameters"]["Optimizer"]))
        if isinstance(config_dict["Train Parameters"]["Loss Function"], str):
            if config_dict["Train Parameters"]["Loss Function"] == "BCEWithLogitsLoss":
                config_dict["Train Parameters"]["Loss Function"] = nn.BCEWithLogitsLoss
            else:
                raise ValueError("The parameter {} for the Loss Function is not defined".format(
                                    config_dict["Train Parameters"]["Loss Function"]))
        return config_dict    

class Files_Module(UserDict):

    def __init__(self, outputFolder:str, mode:str):
        outputFolder = os.path.abspath(outputFolder)
        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        if mode == "Prediction":
            mode_out = "{}/predictionPF2".format(outputFolder)
        elif mode == "Train":
            mode_out = "{}/trainPF2".format(outputFolder)
        elif mode == "Test":
            mode_out = "{}/testPF2".format(outputFolder)
        elif mode == "Map_Embeddings":
            mode_out = "{}/mapembedPF2".format(outputFolder)
        elif mode == "Infere":
            mode_out = "{}/inferePF2".format(outputFolder)
        elif mode == "Align_Proteins":
            mode_out = "{}/alnprotPF2".format(outputFolder)
        else:
            raise ValueError("The mode {} is not available to produce the base output".format(mode))
        if os.path.isdir(mode_out):
            raise OSError("""The folder {} seems to have been used already as """
                        """the output of a previous PathogenFinder2 {} run. """
                        """Please output the results in a different folder.""".format(
                            outputFolder, mode))
        os.mkdir(mode_out)

        self.data = {"base_folder": mode_out, "data_files":{}, "folders":{}}


    def create_nestedoutput(self, format_seq:str, input_file:[bool,str],
                                multi_file:[bool, str]=False):
        
        input_files, input_names = self.add_input(input_file=input_file, multi_file=multi_file)

        self.data["folders"]["results"] = {}
        self.data["folders"]["log"] = {}

        if format_seq in ["genome", "proteome"]:
            self.data["folders"]["preprocess"] = {}
        
        for seq, base_seq in zip(input_files, input_names):
            self.data["data_files"]["input_sequence"][base_seq] = seq
            
            results_folder_seq = "{}/results_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["results"][base_seq] = results_folder_seq
            os.mkdir(results_folder_seq)
            log_folder_seq = "{}/log_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["log"][base_seq] = log_folder_seq
            os.mkdir(log_folder_seq)

            if format_seq in ["genome", "proteome"]:
                preproc_folder_seq = "{}/preprocess_{}".format(self.data["base_folder"], base_seq)
                self.data["folders"]["preprocess"][base_seq] = preproc_folder_seq
                os.mkdir(preproc_folder_seq)
            else:
                preproc_folder_seq = None

            
            if format_seq == "genome":
                self.data["data_files"]["genome_sequence"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]
                self.data["data_files"]["proteome_sequence"][base_seq] = "{}/PredictedProteins.faa".format(preproc_folder_seq)
            else:
                self.data["data_files"]["genome_sequence"][base_seq] = None
                self.data["data_files"]["proteome_sequence"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]
            if format_seq in ["genome", "proteome"]:
                self.data["data_files"]["embedding_file"][base_seq] = "{}/ProteinEmbeddings.h5".format(preproc_folder_seq)
            else:
                self.data["data_files"]["proteome_sequence"][base_seq] = None
                self.data["data_files"]["embedding_file"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]
    
    def create_inferenceoutput(self, input_file:[bool,str], multi_file:[bool, str]=False):

        input_files, input_names = self.add_input(input_file=input_file, multi_file=multi_file)
        self.data["folders"]["files"] = {}
        self.data["folders"]["log"] = {}

        for seq, base_seq in zip(input_files, input_names):
            self.data["data_files"]["input_sequence"][base_seq] = seq

            results_folder_seq = "{}/files_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["files"][base_seq] = results_folder_seq
            os.mkdir(results_folder_seq)
            log_folder_seq = "{}/log_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["log"][base_seq] = log_folder_seq
            os.mkdir(log_folder_seq)

            self.data["data_files"]["genome_sequence"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]
            self.data["data_files"]["proteome_sequence"][base_seq] = "{}/PredictedProteins.faa".format(results_folder_seq)
            self.data["data_files"]["embedding_file"][base_seq] = "{}/ProteinEmbeddings.h5".format(results_folder_seq)


    def add_input(self, input_file:[bool,str], multi_file:[bool, str]=False):
        if not input_file and not multi_file:
            raise ValueError("Input_file or multifile is required")
        elif multi_file and input_file:
            raise ValueError("Input_file and multifile are excluding options")
        elif multi_file:
            input_files, input_names = read_multifiles(multi_file)
        else:
            input_files = [os.path.abspath(input_file)]
            input_names = ["PF2"]
        self.data["input_files"] = input_names
        self.data["input_paths"] = input_files
        self.data["data_files"]["input_sequence"] = {}
        self.data["data_files"]["genome_sequence"] = {}
        self.data["data_files"]["proteome_sequence"] = {}
        self.data["data_files"]["embedding_file"] = {}

        return input_files, input_names

    def add_nn_products(self, produce_embeddings, produce_attentions):
        if produce_embeddings:
            self.data["data_files"]["genome_embeddings"] = {}
        if produce_attentions:
            self.data["data_files"]["attention_file"] = {}        

        for base_seq in self.data["input_files"]:
            if produce_embeddings:
                self.data["data_files"]["genome_embeddings"][base_seq] = "{}/embeddings.npz".format(self.data["folders"]["results"][base_seq])
            if produce_attentions:
                self.data["data_files"]["attention_file"][base_seq] = "{}/attentions.npz".format(self.data["folders"]["results"][base_seq])
            
    
    def postprocess_output(self):
        self.data["folders"]["postprocess"] = {}
        for base_seq in self.data["input_files"]:
            postproc_folder_seq = "{}/postprocess_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["postprocess"][base_seq] = postproc_folder_seq
            os.mkdir(postproc_folder_seq)

    def save_inputmetadata(self, success_proteins:[dict, bool]=False):
        input_files = []
        genome_files = []
        protein_files = []
        embedding_files = []
        for file_base in self.data["input_files"]:
            if success_proteins and success_proteins[file_base]!= "Success":
                continue
            input_files.append(self.data["data_files"]["input_sequence"][file_base])
            genome_files.append(self.data["data_files"]["genome_sequence"][file_base])
            protein_files.append(self.data["data_files"]["proteome_sequence"][file_base])
            embedding_files.append(self.data["data_files"]["embedding_file"][file_base])

        metadata = pd.DataFrame({"Input_Files":input_files, "File_Genome":genome_files,
                            "File_Proteins":protein_files, "File_Embedding":embedding_files})
        self.data["input_metadata"]="{}/data_input.tsv".format(self.data["base_folder"])
        metadata.to_csv(self.data["input_metadata"], sep="\t", index=False)
        return metadata

    


