import sys
import os
import json
from torch import nn
import torch
import argparse
import types
from collections import UserDict

from utils import is_valid_file

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
            raise KeyError("The parameter {} is not here.".format(param))
        else:
            if isinstance(value, str) and value in ParamsModel.function_param:
                function = ParamsModel.function_param[value]
                self[param] = function
            elif not type_d:
                self[param] = value
            else:
                self[param] = type_d(value)
        


class ConfigModel:

    def __init__(self, model_type="additive"):

        self.model_type = model_type
        
        self.misc_parameters = self.init_misc_parameters()
        self.model_parameters = self.init_model_parameters()
        self.train_parameters = self.init_train_parameters()

    def init_misc_parameters(self):
        misc_parameters = ParamsModel(name_params="Misc Parameters")
        misc_parameters.init_params(
                list_params=["Notes"])
        return misc_parameters

    def init_model_parameters(self):
        model_parameters = ParamsModel(name_params="Model Parameters")
        model_parameters.init_params(
                list_params=["attention_type", "in_dropout", "n_conv_l", "kernels",
                    "in_conv_dim", "conv_dropout", "conv_act", "conv_init",
                    "att_in_dim", "att_size", "att_init_hid", "att_act",
                    "att_init_layer", "att_dropout", "fnn_layers", "fnn_dim",
                    "fnn_act", "fnn_init", "fnn_dropout", "out_dim", "out_init", "out_sigmoid",
                    "batch_norm", "layer_norm", "train_status", "saved_model",
                    "mode"])
        return model_parameters

    def init_train_parameters(self):
        train_parameters = ParamsModel(name_params="Train Parameters")
        train_parameters.init_params(
                list_params=["batch_size", "optimizer", "learning_rate",
                    "epochs", "imbalance_sample", "imbalance_weight", "lr_scheduler",
                    "weight_decay", "lr_end", "fused_OptBack",
                    "mix_prec", "asynchronity", "data_sample", "cluster_tsv", "prot_dim_split",
                    "loss_function", "train_df", "train_loc", "val_df", "val_loc",
                    "train_results", "memory_report", "results_dir"])
        return train_parameters

    def standard_init_model(self):

        self.model_parameters.set_param(param="conv_act", value="ReLU", type_d="function")
        self.model_parameters.set_param(param="conv_init", value="kaiming_init", type_d="function")
        self.model_parameters.set_param(param="attention_type", value=self.model_type, type_d=str)
        self.model_parameters.set_param(param="att_init_hid", value="xavier_init", type_d="function")
        self.model_parameters.set_param(param="att_act", value="Tanh", type_d="function")
        self.model_parameters.set_param(param="att_init_layer", value="xavier_init", type_d="function")
        self.model_parameters.set_param(param="fnn_act", value="LeakyReLU", type_d="function")
        self.model_parameters.set_param(param="fnn_init", value="kaiming_init", type_d="function")
        self.model_parameters.set_param(param="out_init", value="xavier_init", type_d="function")
        self.model_parameters.set_param(param="saved_model", value=False, type_d=bool)

    def standard_init_train(self):
#        self.train_parameters.set_param(param="loss_function", value="BCELoss", type_d="function")
 #       self.train_parameters.set_param(param="optimizer", value="Adam", type_d="function")
        pass
    
    def load_model_params(self, args):
        self.model_parameters.set_param(param="n_conv_l", value=args.conv_layers, type_d=int)
        self.model_parameters.set_param(param="kernels", value=args.kernels, type_d=list)
        self.model_parameters.set_param(param="in_conv_dim", value=args.conv_dim, type_d=list)
        self.model_parameters.set_param(param="conv_dropout", value=args.conv_dropout, type_d=float)
        self.model_parameters.set_param(param="att_in_dim", value=args.attention_dim_in, type_d=int)
        self.model_parameters.set_param(param="att_size", value=args.attention_size, type_d=int)
        self.model_parameters.set_param(param="att_dropout", value=args.attention_dropout, type_d=float)
        self.model_parameters.set_param(param="fnn_layers", value=args.fnn_layers, type_d=int)
        self.model_parameters.set_param(param="fnn_dim", value=args.fnn_hidden, type_d=int)
        self.model_parameters.set_param(param="fnn_dropout", value=args.fnn_dropout, type_d=float)
        self.model_parameters.set_param(param="out_dim", value=args.out_dim, type_d=int)
        self.model_parameters.set_param(param="in_dropout", value=args.in_dropout, type_d=float)
        self.model_parameters.set_param(param="batch_norm", value=args.batch_norm, type_d=bool)
        self.model_parameters.set_param(param="layer_norm", value=args.layer_norm, type_d=bool)
    
    def load_train_params(self, args):
        self.train_parameters.set_param(param="batch_size", value=args.batch_size, type_d=int)
        self.train_parameters.set_param(param="optimizer", value=args.optimizer, type_d=str)
        self.train_parameters.set_param(param="learning_rate", value=args.learning_rate, type_d=float)
        self.train_parameters.set_param(param="epochs", value=args.epochs, type_d=int)
        self.train_parameters.set_param(param="lr_scheduler", value=args.lr_scheduler)
        self.train_parameters.set_param(param="weight_decay", value=args.weight_decay, type_d=float)
        self.train_parameters.set_param(param="imbalance_weight", value=args.imbalance_weight, type_d=float)
        self.train_parameters.set_param(param="imbalance_sample", value=args.imbalance_sample, type_d=float)
        self.train_parameters.set_param(param="lr_end", value=args.lr_end, type_d=float)
        self.train_parameters.set_param(param="mix_prec", value=args.precision, type_d=bool)
        self.train_parameters.set_param(param="fused_OptBack", value=args.fused_optback, type_d=bool)
        self.train_parameters.set_param(param="asynchronity", value=args.asynchronity, type_d=bool)
        self.train_parameters.set_param(param="data_sample", value=args.data_sample, type_d=bool)
        self.train_parameters.set_param(param="cluster_tsv", value=args.cluster_tsv, type_d=str)
        self.train_parameters.set_param(param="prot_dim_split", value=args.prot_dim_split, type_d=int)
        self.train_parameters.set_param(param="train_df", value=args.train_df, type_d=str)
        self.train_parameters.set_param(param="train_loc", value=args.train_loc, type_d=str)
        self.train_parameters.set_param(param="val_df", value=args.val_df, type_d=str)
        self.train_parameters.set_param(param="val_loc", value=args.val_loc, type_d=str)
        self.train_parameters.set_param(param="train_results", value=args_train_res, type_d=str)
        self.train_parameters.set_param(param="memory_report", value=args.memory_report, type_d=str)

    def load_json_params(self, json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
        for params, params_v in json_data.items():
            for k, val in params_v.items():
                if params == "Model Params":
                    self.model_parameters.set_param(param=k, value=val)       
                elif params == "Train Params":
                    self.train_parameters.set_param(param=k, value=val)
    
    def save_data(self, file_save):
        data_save = {"Model Params": dict(self.model_parameters),
                    "Train Params": dict(self.train_parameters)}
        with open(file_save, 'w') as f:
            json.dump(data_save, f, cls=NNEncoder)

    @staticmethod
    def arguments_model():
        parser = argparse.ArgumentParser(prog='Pathogenfinder2.0 Model',
                            description='Model arguments')
        parser.add_argument("-j_in", "--json_INdata", help="json file with data for the model",
                            metavar="FILE", type=lambda x: is_valid_file(parser, x),
                            default=False)
        parser.add_argument("-train", "--train", help="do training", action="store_true")
        parser.add_argument("-predict", "--predict", help="predict from file",
                            metavar="FILE", 
                            type=lambda x: is_valid_file(parser, x),
                            default=False)
        parser_model = parser.add_argument_group('Model Parameters')
        parser_model.add_argument("-m", "--model_type", help="model type",
                                default="additive")
        parser_model.add_argument("-c_l", "--conv_layers",
                            help="amount convolutional", default=3)
        parser_model.add_argument("-k", "--kernels", help="kernel values",
                            default=[5,3,3],
                            type=lambda s: [int(item) for item in s.split(',')])
        parser_model.add_argument("-c_i", "--conv_dim", help="in_dim_conv",
                            type=lambda s: [int(item) for item in s.split(',')])
        parser_model.add_argument("-c_d", "--conv_dropout",
                            help="convolutional dropout")
        parser_model.add_argument("-a_i", "--attention_dim_in",
                                help="attention input dimension")
        parser_model.add_argument("-a_s", "--attention_size",
                                help="attention size")
        parser_model.add_argument("-a_d", "--attention_dropout",
                                help="attention dropout")
        parser_model.add_argument("-f_l", "--fnn_layers", help="amount fnn",
                                    default=1)
        parser_model.add_argument("-f_h", "--fnn_hidden", help="fnn hidden nodes")
        parser_model.add_argument("-f_d", "--fnn_dropout", help="fnn dropout")
        parser_model.add_argument("-o_dim", "--out_dim", help="out dimension",
                            default=1)
        parser_model.add_argument("-b_n", "--batch_norm", help="batch norm",
                            action="store_true")
        parser_model.add_argument("-l_n", "--layer_norm", help="layer norm",
                            action="store_true")
        parser_model.add_argument("-i_d", "--in_dropout", help="input dropout",
                            )
        parser_model.add_argument("-b", "--batch_size", help="batch_size", type=int
                            )
        parser_train = parser.add_argument_group('Training Parameters')
        parser_train.add_argument("-lr", "--learning_rate", help="learning rate")
        parser_train.add_argument("-e", "--epochs", help="epochs")
        parser_train.add_argument("-sc", "--lr_scheduler", help="scheduler", action="store_true")
        parser_train.add_argument("-w", "--weight_decay", help="weight decay")
        parser_train.add_argument("-lr_e", "--lr_end", help="end value of lr with scheduler")
        parser_train.add_argument("-m_p", "--mixed_precision", help="mixed_precision", action="store_true")
        parser_train.add_argument("-d_s", "--data_sample", help="sample data", action="store_true")
        parser_train.add_argument("-p", "--prot_dim_split", help="sample data", action="store_true")
        parser_train.add_argument("-tr_d", "--train_df", help="train_df")
        parser_train.add_argument("-tr_l", "--train_loc", help="train_loc")
        parser_train.add_argument("-val_d", "--val_df", help="val_df")
        parser_train.add_argument("-val_l", "--val_loc", help="val_loc")
        parser_train.add_argument("-clust", "--cluster_tsv", help="cluster_tsv")
        parser_train.add_argument("-res", "--train_res", help="train results")
        parser_train.add_argument("-mem_p", "--memory_report", help="memory profiling")
        parser_train.add_argument("-imb_w", "--imbalance_weight", help="imbalance weight")
        parser_train.add_argument("-imb_s", "--imbalance_sample", help="imbalance sample")

        return parser.parse_args()
