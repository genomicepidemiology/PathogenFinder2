import json
from torch import nn
import torch
import types
from collections import UserDict


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


