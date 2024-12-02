import sys
import os
import json
from torch import nn
import torch
import argparse
import types
from collections import UserDict

from utils.configuration_utils import NNEncoder, ParamsModel


class ConfigurationPF2:

    def __init__(self, mode):

        self.mode = mode

        self.misc_parameters = self.init_params(param_group="Misc Parameters",
                                    list_params=["Notes", "Name", "Actions", "Report Results", "Project Name", "Results Folder"])
        self.model_parameters = self.init_params(param_group="Model Parameters",
                                    list_params=["Model Name", "Input Dimensions", "Network Structure",
                                            "Out Dimensions", "Norm Scale", "Norm Type", "Data Parameters", "Attention Norm",
                                            "Mixed Precision", "Stochastic Depth Prob", "Sequence Dropout", "Attention Dropout",
                                            "Model Weights", "Batch Size", "Seed", "Stochastic Depth Prob Att", "Memory Report",
                                            "Loss Function", "Network Weights"])
        self.train_parameters = None
        self.test_parameters = None
        self.inference_parameters = None
        self.hyperopt_parameters = None

        self.set_mode_parameters(mode=self.mode)


    def set_mode_parameters(self, mode):
        if mode == "Train":
            self.train_parameters = self.init_params(param_group="Train Parameters",
                                        list_params=["Optimizer Parameters", "Epochs", "Loss Function", "Memory Report",
                                            "Save Model", "Report Results", "Train DF", "Train Loc",
                                            "Validation DF", "Validation Loc"])
        elif mode == "Inference":
            self.inference_parameters = self.init_params(param_group="Inference Parameters",
                                    list_params=["Preprocessing Parameters", "Sequence Format", "Input Data",
                                                    "Input Location", "Multiple Files", "Input Metadata"])
        elif mode == "Test":
            self.test_parameters = self.init_params(param_group="Test Parameters",
                                    list_params=["Input Data", "Label File", "Sequence Format"])
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
        return str(final_dict)
    
    def load_args_params(self, args):
        if args.outputFolder is not None:
            self.misc_parameters.set_param(param="Results Folder", value=args.outputFolder)

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
            json.dump(data_save, f, cls=NNEncoder)


