import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

import models.layers.utils as utils
from models.layers.attention import Attention_Methods
from models.layers.denseblock import Bottleneck

class DenseNet_AddAtt_Net(nn.Module):

    def __init__(self, input_dim, num_of_class, num_blocks, kernel_sizes, nodes_fnn, conv_channels,
                        dropout_in, factor_block, dropout_fnn, dropout_att, attention_dim, norm,
                        attention_type="Bahdanau", stride=1):

        super(DenseNet_AddAtt_Net, self).__init__()
        assert len(kernel_sizes) == len(conv_channels)
        assert num_blocks == len(kernel_sizes)

        if norm == "Batch":
            norm_module = nn.BatchNorm1d
        elif norm == "Layer":
            norm_module = nn.LayerNorm
        else:
            norm_module = False

        self.in_layer = nn.Sequential(OrderedDict([
                ("drop_in", nn.Dropout1d(dropout_in))]))

        self.dense_blocks = nn.ModuleList()

        for n_block in range(num_blocks):
            k_size = kernel_sizes[n_block]
            c_conv = conv_channels[n_block]
            block = Bottleneck(in_dim=input_dim, out_dim=c_conv, factor_bottle=factor_block,
                                kernel=k_size, norm=norm)
            self.dense_blocks.append(block)
            input_dim = c_conv

        self.attention_layer = Attention_Methods(attention_type=attention_type,
                                                        dimensions_in=input_dim,
                                                        attention_dim=attention_dim,
                                                        dropout=dropout_att)
        self.linear_1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(input_dim, nodes_fnn)),
            ("batch_norm", nn.BatchNorm1d(nodes_fnn)),
            ("activation", nn.LeakyReLU()),
            ("dropout", nn.Dropout(dropout_fnn))]))

        self.linear_out = nn.Linear(nodes_fnn, num_of_class)

        self.init_weights()

    def init_weights(self):

        torch.nn.init.xavier_normal_(self.linear_out.weight)
        self.linear_out.bias.data.fill_(0.01)

    def masked_AvgPool(self, x, seq_lengths):
        x = torch.sum(x, 2)
        x = torch.divide(x, seq_lengths)
        return x

    def forward(self, x, seq_lengths):
        x = self.in_layer(x)
        mask = utils.create_mask(seq_lengths=seq_lengths, dimensions_batch=x.shape)
        x = x.permute(0, 2, 1)
        ## Convolutional ##
        for layer in self.dense_blocks:
            x = layer(x)
            x = x.masked_fill(mask, 0)

        x = x.permute(0, 2, 1)
        ## Additive Attention ##
        x, attentions = self.attention_layer(x, mask)
        ## FNN ##
        x = self.linear_1(x)
        x = self.linear_out(x)
        return x, attentions

