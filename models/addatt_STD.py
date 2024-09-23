import torch
import math
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

import models.layers.utils as utils
from models.layers.attention import Attention_Methods


class AddAtt_Net(nn.Module):

    def __init__(self, input_dim, num_of_class, nodes_fnn, dropout_fnn, dropout_att, attention_dim,
                 dropout_in, norm, attention_type="Bahdanau"):
        super(AddAtt_Net, self).__init__()

        if norm == "Batch":
            norm_module = nn.BatchNorm1d
        elif norm == "Layer":
            norm_module = nn.LayerNorm
        else:
            norm_module = False

        self.in_layer = nn.Sequential(OrderedDict([
                ("drop_in", nn.Dropout1d(dropout_in))
            ]))
       
        self.attention_layer = Attention_Methods(attention_type=attention_type,
                                                        attention_dim=attention_dim,
							dimensions_in=input_dim,
                                                        dropout=dropout_att)


        self.linear_1 = OrderedDict()
        self.linear_1["linear"] = nn.Linear(input_dim, nodes_fnn)
        if norm_module:
            self.linear_1["norm"] = norm_module(nodes_fnn)
        self.linear_1["activation"] = nn.LeakyReLU()
        self.linear_1["dropout"] = nn.Dropout(dropout_fnn)
        self.linear_1 = nn.Sequential(self.linear_1)

        self.linear_out = nn.Linear(nodes_fnn, num_of_class)

        self.init_weights()


    def init_weights(self):

        torch.nn.init.kaiming_normal_(self.linear_1[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        self.linear_1[0].bias.data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.linear_out.weight)
        self.linear_out.bias.data.fill_(0.01)

    def forward(self, x, seq_lengths):
        x = self.in_layer(x)
        mask = utils.create_mask(seq_lengths=seq_lengths, dimensions_batch=x.shape)
        ## Additive Attention ##
        x, attentions = self.attention_layer(x, mask)
        ## FNN ##
        x = self.linear_1(x)
        x = self.linear_out(x)
        return x, attentions

