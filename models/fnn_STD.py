import torch
import math
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class FNN_Net(nn.Module):


    def __init__(self, input_dim=1, nodes_fnn=10, dropout_fnn=0.3, num_of_class=1):
        super(FNN_Net, self).__init__()
        
        self.linear_layer = nn.Sequential(OrderedDict([
            ("fnn", nn.Linear(input_dim, nodes_fnn)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_fnn))]))
        self.linear_out = nn.Linear(nodes_fnn, num_of_class)

    def forward(self, x, lengths):
        x = self.linear_layer(x)
        x = self.linear_out(x)
        return x, None

