import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import models.layers.utils as utils

class Conv1D_Net(nn.Module):

    def __init__(self, input_dim, num_of_class, kernel_sizes, conv_channels,
                 dropout_conv, dropout_in, batch_norm, layer_norm=False, stride=1):
        super(Conv1D_Net, self).__init__()

        if batch_norm:
            dropout_conv = 0

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

        self.linear_out = nn.Linear(input_dim, num_of_class)

        self.init_weights()

    def init_weights(self):

        for layer in self.conv1d_layers:
            torch.nn.init.kaiming_normal_(layer[0].weight, mode='fan_in', nonlinearity='relu')
            layer[0].bias.data.fill_(0.01)

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
        for layer in self.conv1d_layers:
            x = layer(x)
            x = x.masked_fill(mask, 0)

        x = self.masked_AvgPool(x, seq_lengths)
        x = torch.squeeze(x, 1)
        x = self.linear_out(x)
        return x, None

