import os
import json
from torch import nn
import types
import torch

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

class NNEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, nn.ReLU) or isinstance(obj, nn.LeakyReLU) or isinstance(obj, nn.Tanh) or isinstance(obj, torch.nn.modules.loss.BCEWithLogitsLoss):
            return str(obj)
        if isinstance(obj, types.FunctionType):
            return obj.__name__
        if obj.__class__.__name__ == "type":
            return obj.__name__
        return super(NNEncoder, self).default(obj)