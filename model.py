import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from tqdm import tqdm
import h5py
import argparse
import pickle
import gc
import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

sys.dont_write_bytecode = True

from models.fnn_STD import FNN_Net
from models.conv1d_addatt_STD import Conv1D_AddAtt_Net
from models.conv1d_STD import Conv1D_Net
from config_model import ConfigModel, ParamsModel
from train_model import Train_NeuralNet
from prediction import Prediction_NeuralNet
from utils import NNUtils
from results_record import Json_Results



class Compile_Model:

    Dim_prot = 1024

    def __init__(self, model_arguments):

        self.config = ConfigModel(model_type=model_arguments.model_type)

        if model_arguments.json_INdata:
            self.load_json(json_file=model_arguments.json_INdata)
        else:
            self.load_model_params(args=model_arguments)
            self.load_train_params(args=model_arguments)

        self.model_name = self.config.model_parameters["model_name"]
        self.model_carcass = self.get_model_carcass(model_type=self.model_name)

        self.set_model()

        self.results_dir = NNUtils.set_results_files(
				results_dir=self.config.train_parameters["results_dir"])


    def get_model_carcass(self, model_type):
        if model_type == "conv1d_additiveatt":
            return Conv1D_AddAtt_Net
        elif model_type == "conv1d":
            return Conv1D_Net
        elif model_type == "fnn":
            return FNN_Net
        else:
            raise ValueError("Only additive is allowed at the moment")

    def set_model(self):
        if self.model_name == "conv1d_additiveatt":
            self.set_model_conv1d_additiveatt()
        elif self.model_name == "conv1d":
            self.set_model_conv1d()
        elif self.model_name == "fnn":
            self.set_model_fnn()
        else:
            raise KeyError("The model {} is not settled".format(self.model_name))

    def set_model_fnn(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        nodes_fnn=self.config.model_parameters["model_structure"]["fnn_dim"],
                        dropout_fnn=self.config.model_parameters["model_structure"]["fnn_dropout"])

    def set_model_conv1d(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        conv_channels=self.config.model_parameters["model_structure"]["conv_channels"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        kernel_sizes=self.config.model_parameters["model_structure"]["kernels"], stride=1,
                        dropout_conv=self.config.model_parameters["model_structure"]["conv_dropout"],
                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
                        batch_norm=self.config.model_parameters["batch_norm"],
                        layer_norm=self.config.model_parameters["layer_norm"])        

    def set_model_conv1d_additiveatt(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        conv_channels=self.config.model_parameters["model_structure"]["conv_channels"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        kernel_sizes=self.config.model_parameters["model_structure"]["kernels"], stride=1,
                        nodes_fnn=self.config.model_parameters["model_structure"]["fnn_dim"],
                        dropout_conv=self.config.model_parameters["model_structure"]["conv_dropout"],
                        dropout_fnn=self.config.model_parameters["model_structure"]["fnn_dropout"],
                        dropout_att=self.config.model_parameters["model_structure"]["att_dropout"],
                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
                        batch_norm=self.config.model_parameters["batch_norm"],
                        layer_norm=self.config.model_parameters["layer_norm"])



    def load_model_params(self, args):
        self.config.standard_init_model()
        self.config.load_model_params(args)

    def load_train_params(self, args):
        self.config.standard_init_train()
        self.config.load_train_params(args)

    def load_json(self, json_file):
        self.config.standard_init_model()
        self.config.standard_init_train()
        self.config.load_json_params(json_file)

    def save_model(self, path, state_dict=True):
        if state_dict:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, PATH)
    
    def load_model(self, path, state_dict=True):
        if state_dict:
            self.model_carcass.load_state_dict(torch.load(path))
        else:
            self.model = torch.load(path)
    
    def get_config(self):
        return self.config
    
    def get_model(self):
        return self.model

    def predict(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

    def is_dual(self):
        if self.config.model_parameters["out_dim"] == 2:
            dual_pred = True
        else:
            dual_pred = False
        return dual_pred

    def create_hp_report(self):
        params_str = {v: k for k, v in ParamsModel.function_param.items()}
        hp_model = {}
        for k, v in self.config.model_parameters:
            if v in params_str:
                hp_model[k] = params_str[v]
            else:
                hp_model[k] = str(v)
        hp_train = {}
        for k, v in self.config.train_parameters:
            if isinstance(v, list):
                hp_train[k] = ", ".join(v)
            elif v in params_str:
                hp_train[k] = params_str[v]
            else:
                hp_train[k] = str(v)
        return hp_model, hp_train
        
    def train_model(self, report="dictionary"):
        if self.model is None:
            raise ValueError("Set the Model First")
        if report not in ["dictionary", "wandb", "tensorboard"]:
            raise KeyError("The type of report {} is not available".format(report))

        dual_pred = self.is_dual()
        # Define Model
        train_instance = Train_NeuralNet(network=self.model, configuration=self.config,
                            loss_function=self.config.train_parameters["loss_function"],
                            results_dir=self.results_dir,
                            memory_report=self.config.train_parameters["memory_report"],
                            mixed_precision=self.config.train_parameters["mix_prec"],
                            compiler=self.config.train_parameters["compiler"],
			    wandb_report=self.config.train_parameters["wandb_report"])
        # Create Train data
        train_instance.create_dataset(data_df=self.config.train_parameters["train_df"],
                            data_loc=self.config.train_parameters["train_loc"],
                            data_type="train",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred,
                            weighted=self.config.train_parameters["imbalance_weight"],
                            normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"],
                            limit_length=self.config.train_parameters["limit_length"])
        # Create Val data
        train_instance.create_dataset(data_df=self.config.train_parameters["val_df"],
                            data_loc=self.config.train_parameters["val_loc"],
                            data_type="validation",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred, normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])

        # Train Model
        self.config.model_parameters["train_status"] = "Start"
        best_model = train_instance.train(epochs=self.config.train_parameters["epochs"],
                            batch_size=self.config.model_parameters["batch_size"],
                            optimizer=self.config.train_parameters["optimizer"],
                            learning_rate=self.config.train_parameters["learning_rate"], 
                            weight_decay=self.config.train_parameters["weight_decay"],
                            lr_schedule=self.config.train_parameters["lr_scheduler"],
                            end_lr=self.config.train_parameters["lr_end"],
                            amsgrad=False, num_workers=self.config.train_parameters["num_workers"],
                            asynchronity=self.config.train_parameters["asynchronity"],
                            clipping=self.config.train_parameters["clipping"],
                            bucketing=self.config.train_parameters["bucketing"],
                            stratified=self.config.train_parameters["stratified"],
                            warmup_period=self.config.train_parameters["warm_up"])
     

        if self.config.model_parameters["saved_model"]:
            self.save_model(self.config.model_parameters["saved_model"])
        self.config.model_parameters["train_status"]

        train_instance.save_model(best_model)

        return train_instance.results_dir

    def predict_model(self, data):
        if self.model is None:
            raise ValueError("Set the Model First")
        if report not in ["dictionary", "wandb", "tensorboard"]:
            raise KeyError("The type of report {} is not available".format(report))

        pred_instance.create_dataset(data_df=self.config.train_parameters["val_df"],
                            data_loc=self.config.train_parameters["val_loc"],
                            data_type="validation",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred, normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])



if __name__ == "__main__":
    model_arguments = ConfigModel.arguments_model()

    compiled_model = Compile_Model(model_arguments=model_arguments)
    if model_arguments.json_INdata:
        compiled_model.load_json(json_file=model_arguments.json_INdata)
    else:
        compiled_model.load_model_params(args=model_arguments)
        compiled_model.load_train_params(args=model_arguments)

    config = compiled_model.get_config()

    if model_arguments.train:
        results_dir_train = compiled_model.train_model()
        config.save_data("{}/config_file.json".format(results_dir_train))
    if model_arguments.predict:
        compiled_model.predict_model()
