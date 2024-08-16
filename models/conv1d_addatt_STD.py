import torch
import math
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

import models.layers.utils as utils
from models.layers.attention import Attention_Methods


class Conv1D_AddAtt_Net(nn.Module):

    def __init__(self, input_dim, conv_channels, num_of_class, kernel_sizes, nodes_fnn, attention_dim,
                 dropout_conv, dropout_fnn, dropout_att, dropout_in, batch_norm, 
                 layer_norm=False, attention_type="Bahdanau", stride=1):
        super(Conv1D_AddAtt_Net, self).__init__()
        
        assert len(kernel_sizes) == len(conv_channels)
        self.in_layer = nn.Sequential(OrderedDict([
                ("drop_in", nn.Dropout1d(dropout_in))]))

        self.conv1d_layers = nn.ModuleList()
        
        for kernel, dim, in zip(kernel_sizes, conv_channels):
            layer_conv = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(input_dim, dim,
                        kernel_size=kernel,
                        stride=1, padding=kernel//2)),
            ("batch_norm", nn.BatchNorm1d(dim)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))
            self.conv1d_layers.append(layer_conv)
            input_dim = dim
            
            if not batch_norm:
                del layer_conv[1]
       
        self.attention_layer = Attention_Methods(attention_type=attention_type,
							dimensions_in=conv_channels[-1],
                                                        attention_dim=attention_dim,
							dropout=dropout_att)
        self.linear_1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(conv_channels[-1], nodes_fnn)),
            ("batch_norm", nn.BatchNorm1d(nodes_fnn)),
            ("activation", nn.LeakyReLU()),
            ("dropout", nn.Dropout(dropout_fnn))]))

        self.linear_out = nn.Linear(nodes_fnn, num_of_class)

        self.init_weights()
        if not batch_norm: 
            del self.linear_1[1]

    def init_weights(self):
        for layer in self.conv1d_layers:
            torch.nn.init.kaiming_normal_(layer[0].weight, mode='fan_in', nonlinearity='relu')
            layer[0].bias.data.fill_(0.01)

        torch.nn.init.kaiming_normal_(self.linear_1[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        self.linear_1[0].bias.data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.linear_out.weight)
        self.linear_out.bias.data.fill_(0.01)

    def forward(self, x, seq_lengths):
        x = self.in_layer(x)
        mask = utils.create_mask(seq_lengths=seq_lengths, dimensions_batch=x.shape)

        x = x.permute(0, 2, 1)

        ## Convolutional ##
        for layer in self.conv1d_layers:
            x = layer(x)
            x = x.masked_fill(mask, 0)

        x = x.permute(0, 2, 1)
        ## Additive Attention ##
        x, attentions = self.attention_layer(x, mask)
        ## FNN ##
        x = self.linear_1(x)
        x = self.linear_out(x)
        return x, attentions

