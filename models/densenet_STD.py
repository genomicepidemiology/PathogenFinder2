import torch
import math
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

import models.layers.utils as utils
from models.layers.denseblock import Bottleneck


class DenseNet_Net(nn.Module):

    def __init__(self, input_dim, num_of_class, num_blocks, kernel_sizes,
			conv_channels, dropout_in, factor_block, stride=1):
        super(DenseNet_Net, self).__init__()
        assert len(kernel_sizes) == len(conv_channels)
        assert num_blocks == len(kernel_sizes)

        self.in_layer = nn.Sequential(OrderedDict([
                ("drop_in", nn.Dropout1d(dropout_in))]))

        self.dense_blocks = nn.ModuleList()

        for n_block in range(num_blocks):
            k_size = kernel_sizes[n_block]
            c_conv = conv_channels[n_block]
            block = Bottleneck(in_dim=input_dim, out_dim=c_conv, factor_bottle=factor_block,
				kernel=k_size)
            self.dense_blocks.append(block)
            input_dim = c_conv

        self.linear_out = nn.Linear(input_dim, num_of_class)

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

        x = self.masked_AvgPool(x, seq_lengths)
        x = torch.squeeze(x, 1)
        x = self.linear_out(x)
        return x, None

