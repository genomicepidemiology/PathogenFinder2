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
from torch.optim import swa_utils


sys.dont_write_bytecode = True

from models.fnn_STD import FNN_Net
from models.conv1d_addatt_STD import Conv1D_AddAtt_Net
from models.conv1d_STD import Conv1D_Net
from models.addatt_STD import AddAtt_Net
from models.densenet_STD import DenseNet_Net
from models.densenet_addatt_STD import DenseNet_AddAtt_Net
from models.convnext_STD import ConvNeXt_Net
from models.convnext_addatt_STD import ConvNet_AddAtt_Net


from config_model import ConfigModel, ParamsModel
from train_model import Train_NeuralNet
from prediction import Prediction_NeuralNet
from test_model import Test_NeuralNet
from utils import NNUtils, Metrics
from results_record import Json_Results, Wandb_Results
from data_utils import NN_Data



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
        print(self.model)

        self.results_dir = NNUtils.set_results_files(
				results_dir=self.config.train_parameters["results_dir"])
        self.results_model = Wandb_Results(configuration=self.config,
                                            name=os.path.basename(self.results_dir))

    def get_model_carcass(self, model_type):
        if model_type == "conv1d_additiveatt":
            return Conv1D_AddAtt_Net
        elif model_type == "conv1d":
            return Conv1D_Net
        elif model_type == "fnn":
            return FNN_Net
        elif model_type == "additiveatt":
            return AddAtt_Net
        elif model_type == "densenet":
            return DenseNet_Net
        elif model_type == "densenet_additiveatt":
            return DenseNet_AddAtt_Net
        elif model_type == "convnext":
            return ConvNeXt_Net
        elif model_type == "convnext_additiveatt":
            return ConvNet_AddAtt_Net
        else:
            raise ValueError("Only additive is allowed at the moment")

    def set_model(self):
        if self.model_name == "conv1d_additiveatt":
            self.set_model_conv1d_additiveatt()
        elif self.model_name == "conv1d":
            self.set_model_conv1d()
        elif self.model_name == "additiveatt":
            self.set_model_additiveatt()
        elif self.model_name == "fnn":
            self.set_model_fnn()
        elif self.model_name == "densenet":
            self.set_model_densenet()
        elif self.model_name == "densenet_additiveatt":
            self.set_model_densenet_additiveatt()
        elif self.model_name == "convnext":
            self.set_model_convnext()
        elif self.model_name == "convnext_additiveatt":
            self.set_model_convnext_additiveatt()
        else:
           raise KeyError("The model {} is not settled".format(self.model_name))

    def set_model_convnext(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                                    num_classes=self.config.model_parameters["out_dim"],
                                    num_blocks=self.config.model_parameters["model_structure"]["num_blocks"],
                                    block_dims=self.config.model_parameters["model_structure"]["block_dims"],
                                    downsample=self.config.model_parameters["model_structure"]["downsample"],
                                    stochastic_depth_prob=self.config.train_parameters["stochastic_depth_prob"],
                                    layer_scale=self.config.train_parameters["norm_scale"],
                                    norm=self.config.model_parameters["norm"],
                                    length_information=self.config.model_parameters["length_information"],
                                    length_dim=self.config.model_parameters["length_dim"])
#        self.model = swa_utils.AveragedModel(self.model)

    def set_model_convnext_additiveatt(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                                    num_classes=self.config.model_parameters["out_dim"],
                                    num_blocks=self.config.model_parameters["model_structure"]["num_blocks"],
                                    block_dims=self.config.model_parameters["model_structure"]["block_dims"],
                                    stochastic_depth_prob=self.config.train_parameters["stochastic_depth_prob"],
                                    attention_dim=self.config.model_parameters["model_structure"]["att_dim"],
                                    attention_norm=self.config.model_parameters["model_structure"]["att_norm"],
                                    dropout_att=self.config.model_parameters["model_structure"]["att_dropout"],
                                    layer_scale=self.config.train_parameters["norm_scale"],
                                    sequence_dropout=self.config.model_parameters["sequence_dropout"],
                                    norm=self.config.model_parameters["norm"],
                                    length_information=self.config.model_parameters["length_information"],
                                    length_dim=self.config.model_parameters["length_dim"])
 #       self.model = swa_utils.AveragedModel(self.model)


    def set_model_densenet_additiveatt(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                                        num_of_class=self.config.model_parameters["out_dim"],
                                        num_blocks=self.config.model_parameters["model_structure"]["num_blocks"],
					kernel_sizes=self.config.model_parameters["model_structure"]["kernels"],
		                        conv_channels=self.config.model_parameters["model_structure"]["conv_channels"],
                                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
					factor_block=self.config.model_parameters["model_structure"]["factor_block"],
		                        nodes_fnn=self.config.model_parameters["model_structure"]["fnn_dim"],
                		        dropout_fnn=self.config.model_parameters["model_structure"]["fnn_dropout"],
		                        dropout_att=self.config.model_parameters["model_structure"]["att_dropout"],
		                        attention_dim=self.config.model_parameters["model_structure"]["att_dim"],
		                        norm=self.config.model_parameters["norm"])


    def set_model_densenet(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                                        num_of_class=self.config.model_parameters["out_dim"],
                                        num_blocks=self.config.model_parameters["model_structure"]["num_blocks"],
                                        kernel_sizes=self.config.model_parameters["model_structure"]["kernels"],
                                        conv_channels=self.config.model_parameters["model_structure"]["conv_channels"],
                                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
                                        factor_block=self.config.model_parameters["model_structure"]["factor_block"],
                                        norm=self.config.model_parameters["norm"])


    def set_model_additiveatt(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        nodes_fnn=self.config.model_parameters["model_structure"]["fnn_dim"],
                        dropout_fnn=self.config.model_parameters["model_structure"]["fnn_dropout"],
                        dropout_att=self.config.model_parameters["model_structure"]["att_dropout"],
                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
                        attention_dim=self.config.model_parameters["model_structure"]["att_dim"],
                        norm=self.config.model_parameters["norm"])

    def set_model_fnn(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        nodes_fnn=self.config.model_parameters["model_structure"]["fnn_dim"],
                        dropout_fnn=self.config.model_parameters["model_structure"]["fnn_dropout"],
                        norm=self.config.model_parameters["norm"])

    def set_model_conv1d(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        conv_channels=self.config.model_parameters["model_structure"]["conv_channels"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        kernel_sizes=self.config.model_parameters["model_structure"]["kernels"], stride=1,
                        dropout_conv=self.config.model_parameters["model_structure"]["conv_dropout"],
                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
                        norm=self.config.model_parameters["norm"])        

    def set_model_conv1d_additiveatt(self):
        self.model = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                        conv_channels=self.config.model_parameters["model_structure"]["conv_channels"],
                        num_of_class=self.config.model_parameters["out_dim"],
                        kernel_sizes=self.config.model_parameters["model_structure"]["kernels"], stride=1,
                        nodes_fnn=self.config.model_parameters["model_structure"]["fnn_dim"],
                        attention_dim=self.config.model_parameters["model_structure"]["att_dim"],
                        dropout_conv=self.config.model_parameters["model_structure"]["conv_dropout"],
                        dropout_fnn=self.config.model_parameters["model_structure"]["fnn_dropout"],
                        dropout_att=self.config.model_parameters["model_structure"]["att_dropout"],
                        dropout_in=self.config.model_parameters["model_structure"]["in_dropout"],
                        norm=self.config.model_parameters["norm"])



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

    def save_model(self, data, path, type_save="model"):
        PATH = "{}/model.pt".format(path)
        if type_save == "state_dict":
            torch.save(data.state_dict(), PATH)
        elif type_save == "checkpoint":
            torch.save(data, PATH)
        else:
            torch.save(data, PATH)
    
    def load_model(self, path, type_load="model"):
        PATH = "{}/model.pt".format(path)
        if type_load == "state_dict":
            self.model.load_state_dict(torch.load(PATH))
        elif type_load == "checkpoint":
            self.model.load_state_dict(torch.load(PATH)["model_state_dict"])
        else:
            self.model = torch.load(PATH)
    
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
                            swa_iter=self.config.train_parameters["swa"],
                            memory_report=self.config.train_parameters["memory_report"],
                            mixed_precision=self.config.train_parameters["mix_prec"],
                            compiler=self.config.train_parameters["compiler"],
			    wandb_report=self.results_model)
        # Create Train data
        train_dataset = NN_Data.create_dataset(input_type=self.config.model_parameters["input_type"],
                            data_df=self.config.train_parameters["train_df"],
                            data_loc=self.config.train_parameters["train_loc"],
                            data_type="train",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred,
                            weighted=self.config.train_parameters["imbalance_weight"],
                            normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])
        # Create Val data
        val_dataset = NN_Data.create_dataset(input_type=self.config.model_parameters["input_type"],
                            data_df=self.config.train_parameters["val_df"],
                            data_loc=self.config.train_parameters["val_loc"],
                            data_type="prediction",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred, normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])

        # Train Model
        self.config.model_parameters["train_status"] = "Start"
        best_model = train_instance(train_dataset=train_dataset, val_dataset=val_dataset,
                            epochs=self.config.train_parameters["epochs"],
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
                            warmup_period=self.config.train_parameters["warm_up"],
                            early_stopping=self.config.train_parameters["early_stopping"],
                            keep_model=self.config.train_parameters["save_model"])

        self.config.model_parameters["train_status"] = "Done"

        if self.config.train_parameters["save_model"]:
            self.save_model(data=best_model, path=self.results_dir, type_save="checkpoint")

        if best_model:
            return best_model
        else:
            return False

    def predict_model(self):
        if self.model is None:
            raise ValueError("Set the Model First")

        pred_dataset = NN_Data.create_dataset(input_type=self.config.model_parameters["input_type"],
                            data_df=self.config.predict_parameters["data_df"],
                            data_loc=self.config.predict_parameters["data_loc"],
                            data_type="prediction",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred, normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])
        prediction_instance = Prediction_NeuralNet()
        prediction_instance(pred_dataset=pred_dataset)

    def test_model(self):
        if self.model is None:
            raise ValueError("Set the Model First")

        dual_pred = self.is_dual()

        pred_dataset = NN_Data.create_dataset(input_type=self.config.model_parameters["input_type"],
                            data_df=self.config.test_parameters["test_df"],
                            data_loc=self.config.test_parameters["test_loc"],
                            data_type="prediction",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred, normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])
        test_instance = Test_NeuralNet(network=self.model, configuration=self.config,
                            mixed_precision=self.config.train_parameters["mix_prec"], results_dir=self.results_dir,
                            results_module=self.results_model)
        if self.model_name == "conv1d_additiveatt" or self.model_name == "additiveatt" or self.model_name == "densenet_additiveatt":
            report_att = True
        else:
            report_att = False

        test_instance(test_dataset=pred_dataset, asynchronity=self.config.train_parameters["asynchronity"],
                            num_workers=self.config.train_parameters["num_workers"],
                            bucketing=self.config.train_parameters["bucketing"],
                            stratified=self.config.train_parameters["stratified"],
                            batch_size=self.config.model_parameters["batch_size"], report_att=report_att, return_layer="avgpool")



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
        best_model = compiled_model.train_model()
        config.save_data("{}/config_file.json".format(compiled_model.results_dir))
    if model_arguments.predict:
        compiled_model.load_model(compiled_model.results_dir)
        compiled_model.predict_model()
    if model_arguments.test:
 #       compiled_model.load_model(compiled_model.results_dir, type_load="checkpoint")
        compiled_model.load_model("/work3/alff/results_pathogenfinder2/ConvNextAtt_normlayafter_lr3_07-10-2024_17-34-51", type_load="checkpoint")
   #     compiled_model.load_model("/ceph/hpc/data/d2023d12-072-users/results_training_foolaround/all_data/convnext512126_addatt512Model2D_01-10-2024_18-01-58", type_load="checkpoint")
 #       compiled_model.load_model("/ceph/hpc/data/d2023d12-072-users/results_training_foolaround/all_data/convnext_test_18-09-2024_17-07-53", type_load="checkpoint")
        compiled_model.test_model()

