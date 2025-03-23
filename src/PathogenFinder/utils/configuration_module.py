import sys
import os
import json

import argparse
import types
from pathlib import Path
from collections import UserDict

from ..utils.os_utils import create_outputfolder
from .configuration_utils import NNEncoder, ParamsModel


class ConfigurationPF2:

    def __init__(self, mode:str, user_config:tuple[str,dict]):

        self.mode = mode
        self.inference_parameters = None
        self.train_parameters = None
        self.test_parameters = None
        self.hyperopt_parameters = None

        if isinstance(user_config, str):

            self.misc_parameters = self.init_params(param_group="Misc Parameters",
                                    list_params=["Notes", "Name", "Actions", "Report Results", "Project Name", "Results Folder"])
            self.model_parameters = self.init_params(param_group="Model Parameters",
                                    list_params=["Model Name", "Input Dimensions", "Network Structure",
                                            "Out Dimensions", "Norm Scale", "Norm Type", "Data Parameters", "Attention Norm",
                                            "Mixed Precision", "Stochastic Depth Prob", "Sequence Dropout", "Attention Dropout",
                                            "Model Weights", "Batch Size", "Seed", "Stochastic Depth Prob Att", "Memory Report",
                                            "Loss Function", "Network Weights"])
            self.set_mode_parameters(mode=self.mode)

        else:
            std_json_path = "{}/../../../data/configs/config_empty.json".format(Path(__file__).parent.absolute())
            with open(std_json_path, "r") as stdjson:
                std_json = json.load(stdjson)
            self.misc_parameters = std_json["Misc Parameters"]
            self.model_parameters = std_json["Model Parameters"]
            if mode == "Inference":
                self.inference_parameters = std_json["Inference Parameters"]
            elif mode == "Train":
                self.train_parameters = std_json["Train Parameters"]
            elif mode == "Test":
                self.test_parameters = std_json["Test Parameters"]
            elif mode == "Hyperparam_Opt":
                self.hyperopt_parameters = std_json["Hyperparam_Opt Parameters"]
            else:
                raise ValueError("The mode {} is not available".format(mode))


    def set_mode_parameters(self, mode):
        if mode == "Train":
            self.train_parameters = self.init_params(param_group="Train Parameters",
                                        list_params=["Optimizer Parameters", "Epochs", "Loss Function", "Memory Report",
                                            "Save Model", "Report Results", "Train DF", "Train Loc",
                                            "Validation DF", "Validation Loc"])
        elif mode == "Inference":
            self.inference_parameters = self.init_params(param_group="Inference Parameters",
                                    list_params=["Preprocessing Parameters", "Sequence Format", "Input Data",
                                                    "Input Location", "Multiple Files", "Input Metadata",
                                                    "Attentions", "Embeddings"])
        elif mode == "Test":
            self.test_parameters = self.init_params(param_group="Test Parameters",
                                    list_params=["Input Data", "Label File", "Sequence Format", "Produce Attentions", "Produce Embeddings"])
        elif mode == "Hyperparam_Opt":
            self.hyperopt_parameters = self.init_params(parm_group="Hyperparam_Opt Parameters",
                                    list_params=["Optimizing Parameters", "Name Study", "Group",
                                            "Num Trials", "Load Study", "Storage", "Min Epochs Count",
                                            "Train DF", "Train Loc", "Validation DF", "Validation Loc",
                                            "Timeout", "Timeout", "Try Parameters", "Add Runs"])
        else:
            raise ValueError("The mode {} is not available".format(mode))


    def init_params(self, param_group, list_params):
        parameters = ParamsModel(name_params=param_group)
        parameters.init_params(list_params=list_params)
        return parameters

    def load_json_params(self, json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
        for params, params_v in json_data.items():
            for k, val in params_v.items():
                if params == "Misc Parameters":
                    model_set = self.misc_parameters
                elif params == "Model Parameters":
                    model_set = self.model_parameters
                elif params == "Train Parameters":
                    model_set = self.train_parameters
                elif params == "Inference Parameters":
                    model_set = self.inference_parameters
                elif params == "Test Parameters":
                    model_set = self.test_parameters
                elif params == "Hyperparam_Opt Parameters":
                    model_set = self.hyperopt_parameters
                else:
                    raise ValueError("The Category '{}' in the JSON faile is not available".format(params))
                if model_set is None:
                    raise KeyError("The Parameters '{}' are in the JSON although the mode is '{}'".format(params, self.mode))
                else:
                    model_set.set_param(param=k, value=val)

    def __str__(self):
        final_dict = self.collect_params()
        out_ = "{"
        for k, v in final_dict.items():
            out_ += "\n\t{}: {}".format(k, v)
        out_ += "\n}"
        return out_
        
    def load_dict_params(self, dict_args):
        if dict_args["outputFolder"]:
            self.misc_parameters["Results Folder"] = create_outputfolder(outpath=os.path.abspath(dict_args["outputFolder"]))
        else:
            raise ValueError("It is necessary to set an output folder with --outputFolder when not using a config file")
        if dict_args["inputData"]:
            self.inference_parameters["Input Data"] = dict_args["inputData"]
        else:
            raise ValueError("It is necessary to set the input file with --inputData when not using a config file")
        if dict_args["formatSeq"]:
            self.inference_parameters["Sequence Format"] = dict_args["formatSeq"]
        else:
            raise ValueError("It is necessary to set what type of sequence with --formatSeq when not using a config file")
        self.inference_parameters["Multiple Files"] = dict_args["multiFiles"]
        self.inference_parameters["Embeddings"] = dict_args["embeddings"]
        self.inference_parameters["Attentions"] = dict_args["attentions"]
        
        self.misc_parameters["Prodigal Path"] = dict_args["prodigalPath"]
        self.misc_parameters["ProtT5 Path"] = dict_args["protT5Path"]
        self.misc_parameters["Diamond Path"] = dict_args["diamondPath"]


        if dict_args["weightsModel"]:
            files_weights = []
            for filew in dict_args["formatSeq"].split(","):
                files_weights.append(filew.strip())
            self.model_parameters["Network Weights"] = files_weights
        else:
            weights_path = "%s/../../../data/models_weights/weights_model{}.pickle" % Path(__file__).parent.absolute()
            files_weights = [weights_path.format(1), weights_path.format(2), weights_path.format(3), weights_path.format(4)]
            self.model_parameters["Network Weights"] = files_weights


    def collect_params(self):
        final_dict = {"Misc Parameters": self.misc_parameters,
                            "Model Parameters": self.model_parameters,
                            "Train Parameters": self.train_parameters,
                            "Inference Parameters": self.inference_parameters,
                            "Test Parameters": self.test_parameters,
                            "Hyperparam Opt": self.hyperopt_parameters}
        return final_dict

    def save_json_params(self):
        final_dict = self.collect_params()
        file_save = "{}/config_run.json".format(self.misc_parameters["Results Folder"])
        with open(file_save, 'w') as f:
            json.dump(final_dict, f, cls=NNEncoder)


