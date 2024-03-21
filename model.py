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

sys.dont_write_bytecode = True

from conv1d_addatt_STD import Conv1D_AddAtt_Net
from config_model import ConfigModel
from train_model import Train_NeuralNet


class Compile_Model:

    Dim_prot = 1024

    def __init__(self, model_type="additive"):

        self.config = ConfigModel(model_type=model_type)
        self.model = None
        self.model_carcass = self.get_model_carcass(model_type=model_type)

    def get_model_carcass(self, model_type):
        if model_type == "additive":
            return Conv1D_AddAtt_Net
        else:
            raise ValueError("Only additive is allowed at the moment")

    def set_model(self):
        self.model = self.model_carcass(
                        conv_in_features=self.config.model_parameters["in_conv_dim"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        kernel_sizes=self.config.model_parameters["kernels"], stride=1,
                        conv_out_dim=self.config.model_parameters["att_in_dim"],
                        nodes_fnn=self.config.model_parameters["fnn_dim"],
                        dropout_conv=self.config.model_parameters["conv_dropout"],
                        att_size=self.config.model_parameters["att_size"],
                        dropout_fnn=self.config.model_parameters["fnn_dropout"],
                        dropout_att=self.config.model_parameters["att_dropout"],
                        dropout_in=self.config.model_parameters["in_dropout"],
                        batch_norm=self.config.model_parameters["batch_norm"],
                        layer_norm=self.config.model_parameters["layer_norm"],
                        act_fnn=self.config.model_parameters["fnn_act"],
                        act_conv=self.config.model_parameters["conv_act"])


        Conv1D_AddAtt_Net.init_weights(module=self.model.layer_conv1.conv1d,
                            init_weights=self.config.model_parameters["conv_init"],
                            layer_type="conv", nonlinearity="relu")
        Conv1D_AddAtt_Net.init_weights(module=self.model.layer_conv2.conv1d,
                            init_weights=self.config.model_parameters["conv_init"],
                            layer_type="conv", nonlinearity="relu")
        Conv1D_AddAtt_Net.init_weights(module=self.model.layer_conv3.conv1d,
                            init_weights=self.config.model_parameters["conv_init"],
                            layer_type="conv", nonlinearity="relu")
        Conv1D_AddAtt_Net.init_weights(module=self.model.linear_in_att,
                            init_weights=self.config.model_parameters["att_init_layer"],
                            layer_type="att")
        Conv1D_AddAtt_Net.init_weights(module=self.model.linear_att,
                            init_weights=self.config.model_parameters["att_init_hid"],
                            layer_type="att")
        Conv1D_AddAtt_Net.init_weights(module=self.model.linear_1.linear,
                            init_weights=self.config.model_parameters["fnn_init"],
                            layer_type="fnn", nonlinearity="leaky_relu")

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
    
    def train_model(self):
        if self.model is None:
            raise ValueError("Set the Model First")
        
        if self.config.model_parameters["out_dim"] == 2:
            dual_pred = True
        else:
            dual_pred = False

        train_instance = Train_NeuralNet(network=self.model,
                            learning_rate=self.config.train_parameters["learning_rate"],
                            weight_decay=self.config.train_parameters["weight_decay"],
                            loss_function=self.config.train_parameters["loss_function"],
                            )
    
        train_instance.create_dataset(data_df=self.config.train_parameters["train_df"],
                            data_loc=self.config.train_parameters["train_loc"],
                            data_type="train",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred,
                            weighted=self.config.train_parameters["imbalance_weight"])
        train_instance.create_dataset(data_df=self.config.train_parameters["val_df"],
                            data_loc=self.config.train_parameters["val_loc"],
                            data_type="validation",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred)
    
        self.config.model_parameters["train_status"] = "Start"
        train_res, self.model = train_instance.train(epochs=self.config.train_parameters["epochs"],
                            batch_size=self.config.train_parameters["batch_size"],
                            lr_schedule=self.config.train_parameters["lr_scheduler"],
                            end_lr=self.config.train_parameters["lr_end"],
                            mixed_precision=self.config.train_parameters["mix_prec"],
                            memory_profile=self.config.train_parameters["memory_profile"])
    
        with open(self.config.train_parameters["train_results"], 'wb') as f:
            pickle.dump(train_res, f)
        if self.config.model_parameters["saved_model"]:
            self.save_model(self.config.model_parameters["saved_model"])
        self.config.model_parameters["train_status"] = "Done"

        return True        
        
                  

if __name__ == "__main__":
    model_arguments = ConfigModel.arguments_model()

    compiled_model = Compile_Model(model_type=model_arguments.model_type)
    if model_arguments.json_INdata:
        compiled_model.load_json(json_file=model_arguments.json_INdata)
    else:
        compiled_model.load_model_params(args=model_arguments)
        compiled_model.load_train_params(args=model_arguments)
    compiled_model.set_model()

    if model_arguments.train:
        compiled_model.train_model()
    elif model_arguments.predict:
        pass

    config = compiled_model.get_config()
    config.save_data("./config_demo.json")


