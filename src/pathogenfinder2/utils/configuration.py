import os
import json
import types
import pandas as pd
from torch import nn, optim
import torch

from pathlib import Path
from collections import UserDict
from typing import Union

from pathogenfinder2.utils.io import read_multifiles
from pathogenfinder2.exceptions import ConfigurationError
from pathogenfinder2.utils.config_schema import PREDICTION_SCHEMA, USER_CONFIG_SCHEMA


class NNEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, nn.ReLU) or isinstance(obj, nn.LeakyReLU) or isinstance(obj, nn.Tanh) or isinstance(obj, torch.nn.modules.loss.BCEWithLogitsLoss):
            return str(obj)
        if isinstance(obj, types.FunctionType):
            return obj.__name__
        if obj.__class__.__name__ == "type":
            return obj.__name__
        return super(NNEncoder, self).default(obj)


class ParamsModel(UserDict):

    function_param = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "NAdam": torch.optim.NAdam,
            "BCELoss": nn.BCELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "xavier_init": nn.init.xavier_normal_,
            "kaiming_init": nn.init.kaiming_normal_,
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(),
            "ReLU": nn.ReLU()
            }

    def __init__(self, name_params):
        UserDict.__init__(self)
        self.name_params = name_params

    def init_params(self, list_params):
        for param in list_params:
            self[param] = None

    def set_param(self, param, value, type_d=False):
        if param not in self:
            raise KeyError("The parameter {} is not part of the Config of PF2.".format(param))
        else:
            if isinstance(value, dict):
                self[param] = {}
                for k, val in value.items():
                    if isinstance(val, str) and val in ParamsModel.function_param:
                        function = ParamsModel.function_param[val]
                        self[param][k] = function
                    else:
                        self[param][k] = val
            elif isinstance(value, str) and value in ParamsModel.function_param:
                function = ParamsModel.function_param[value]
                self[param] = function
            elif not type_d:
                self[param] = value
            else:
                self[param] = type_d(value)



class ConfigurationPF2(UserDict):

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, mode: str, user_config: Union[str, dict, bool] = False):

        self.data = {}
        self.mode = mode
        if user_config is False:
            config_data = ConfigurationPF2.load_baseconfig()
            config_data = ConfigurationPF2.load_baseweights(config_data=config_data)
            ConfigurationPF2.validate(config_data, PREDICTION_SCHEMA)
        else:
            config_data = self.load_user_config(user_config=user_config)
            ConfigurationPF2.validate(config_data, USER_CONFIG_SCHEMA)
        config_data["Misc Parameters"]["Actions"] = mode
        self.update(config_data)

    @staticmethod
    def validate(config_data: dict, schema: dict) -> None:
        """Validate *config_data* against *schema*, raising :exc:`ConfigurationError` on failure."""
        try:
            import jsonschema
        except ImportError:
            return  # jsonschema not installed — skip validation silently
        try:
            jsonschema.validate(instance=config_data, schema=schema)
        except jsonschema.ValidationError as exc:
            raise ConfigurationError(
                f"Configuration error at '{'/'.join(str(p) for p in exc.absolute_path)}': {exc.message}"
            ) from exc

    @staticmethod
    def load_baseconfig():
        config_file = Path(ConfigurationPF2.CURRENT_DIR) / '../data/configs/config_base.json'
        with open(config_file) as f:
            config_dict = json.load(f)
        return config_dict
    
    @staticmethod
    def load_baseweights(config_data):
        for i in [1, 2, 3, 4]:
            weight_file = Path(ConfigurationPF2.CURRENT_DIR) / '../data/models_weights/weights_model{}.pt'.format(i)
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
                raise ConfigurationError("The optimizer '{}' is not defined".format(
                                    config_dict["Train Parameters"]["Optimizer"]))
        if isinstance(config_dict["Train Parameters"]["Loss Function"], str):
            if config_dict["Train Parameters"]["Loss Function"] == "BCEWithLogitsLoss":
                config_dict["Train Parameters"]["Loss Function"] = nn.BCEWithLogitsLoss
            else:
                raise ConfigurationError("The loss function '{}' is not defined".format(
                                    config_dict["Train Parameters"]["Loss Function"]))
        return config_dict    

class FilesModule(UserDict):

    def __init__(self, outputFolder:str, mode:str):
        outputFolder = os.path.abspath(outputFolder)
        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        _MODE_DIRS = {
            "Prediction": "predictionPF2",
            "Train": "trainPF2",
            "Test": "testPF2",
            "Map_Embeddings": "mapembedPF2",
            "Infer": "inferPF2",
            "Align_Proteins": "alnprotPF2",
        }
        if mode not in _MODE_DIRS:
            raise ValueError("The mode {} is not available to produce the base output".format(mode))
        mode_out = "{}/{}".format(outputFolder, _MODE_DIRS[mode])
        if os.path.isdir(mode_out):
            raise OSError("""The folder {} seems to have been used already as """
                        """the output of a previous PathogenFinder2 {} run. """
                        """Please output the results in a different folder.""".format(
                            outputFolder, mode))
        os.mkdir(mode_out)

        self.data = {"base_folder": mode_out, "data_files":{}, "folders":{}}


    def create_nestedoutput(self, format_seq: str, input_file: bool | str,
                                multi_file: bool | str = False):
        
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
    
    def create_inferenceoutput(self, input_file: bool | str, multi_file: bool | str = False):

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


    def add_input(self, input_file: bool | str, multi_file: bool | str = False):
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

    def embedding_file(self, key: str) -> str:
        return self.data["data_files"]["embedding_file"][key]

    def proteome_file(self, key: str) -> str:
        return self.data["data_files"]["proteome_sequence"][key]

    def results_folder(self, key: str) -> str:
        return self.data["folders"]["results"][key]

    def log_folder(self, key: str) -> str:
        return self.data["folders"]["log"][key]

    def attention_file(self, key: str) -> str:
        return self.data["data_files"]["attention_file"][key]

    def postprocess_folder(self, key: str) -> str:
        return self.data["folders"]["postprocess"][key]

    def save_inputmetadata(self, success_proteins: dict | None = None):
        input_files = []
        genome_files = []
        protein_files = []
        embedding_files = []
        for file_base in self.data["input_files"]:
            if success_proteins is not None and success_proteins[file_base] != "Success":
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

    


