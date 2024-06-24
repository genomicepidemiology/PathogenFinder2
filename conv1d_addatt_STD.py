import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from tqdm import tqdm
import h5py
import argparse
import pickle
import gc
import os
import pandas as pd
import numpy as np
from collections import OrderedDict

class Conv1D_AddAtt_Net(nn.Module):

    def __init__(self, conv_in_features=[1024, 1024*4, 1024*2], num_of_class=1,
                 kernel_sizes=[5,3,3], stride=1, conv_out_dim=100, nodes_fnn=50,
                 dropout_conv=0.2, att_size=200, dropout_fnn=0.3, dropout_att=0.3,
                 dropout_in=0.4, batch_norm=False, layer_norm=False,
                 act_conv=nn.ReLU(), act_fnn=nn.LeakyReLU(), act_att=nn.Tanh()):
        super(Conv1D_AddAtt_Net, self).__init__()

        if batch_norm:
            self.in_layer = nn.Sequential(OrderedDict([
                ("batchnorm", nn.BatchNorm2d(conv_in_features[0])),
                ("drop_in", nn.Dropout1d(dropout_in))
            ]))
        else:
            self.in_layer = nn.Sequential(OrderedDict([
                ("drop_in", nn.Dropout1d(dropout_in))
            ]))

        self.layer_conv1 = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(conv_in_features[0], conv_in_features[1],
                        kernel_size=kernel_sizes[0],
                        stride=1, padding=kernel_sizes[0]//2)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))

        self.layer_conv2 = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(conv_in_features[1], conv_in_features[2],
                                    kernel_size=kernel_sizes[1],
                                    stride=1, padding=kernel_sizes[1]//2)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))

        self.layer_conv3 = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(conv_in_features[2], conv_out_dim,
                                    kernel_size=kernel_sizes[2],
                                    stride=1, padding=kernel_sizes[2]//2)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))

        self.linear_in_att = nn.Linear(conv_out_dim, att_size)
        self.linear_att = nn.Linear(att_size, 1, bias=False)
        self.att_act = act_att
        self.dropout_att = nn.Dropout(dropout_att)

        self.linear_1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(conv_out_dim, nodes_fnn)),
            ("activation", nn.LeakyReLU()),
            ("dropout", nn.Dropout(dropout_fnn))]))

        self.linear_out = nn.Linear(nodes_fnn, num_of_class)

    @staticmethod
    def init_weights(module, init_weights, layer_type, nonlinearity=None):
        if layer_type == "conv":
            init_weights(module.weight, mode='fan_in', nonlinearity=nonlinearity)
        elif layer_type == "att":
            init_weights(module.weight)
        elif layer_type == "fnn":
            init_weights(module.weight, mode='fan_in', nonlinearity=nonlinearity)            
        else:
            raise ValueError("The layer {} is not possible".format(layer_type))
        
        if module.bias is not None:
            module.bias.data.fill_(0.001)

    def attention_pass(self, x_in, seq_lengths):  # x_in.shape: [bs, seq_len, in_size]
        att_vector = self.linear_in_att(x_in) # [bs, seq_len, att_size]
        att_hid_align = self.att_act(att_vector) # [bs, seq_len, att_size]
        att_hid_align = self.dropout_att(att_hid_align)
        att_score = self.linear_att(att_hid_align).squeeze(2) # [bs, seq_len]
        mask = Conv1D_AddAtt_Net.length_to_negative_mask(seq_lengths)
        alpha = F.softmax(att_score + mask, dim=1) # [bs, seq_len]
        att = alpha.unsqueeze(2) # [bs, seq_len, 1]

        return torch.sum(x_in * att, dim=1), alpha # [bs, in_size]

    @staticmethod
    def length_to_negative_mask(length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        if len(length.shape) != 1: # 'Length shape should be 1 dimensional.'
            length = length.squeeze()

        assert len(length.shape) == 1
        max_len = max_len or int(length.max().item())
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        mask = mask.float()
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        mask = (mask - 1) * 10e6
        return mask

    def forward(self, x, seq_lengths):
        x = self.in_layer(x)
        x = x.permute(0, 2, 1)
        ## Convolutional ##
        x = self.layer_conv1(x)
        x = self.layer_conv2(x)
        x = self.layer_conv3(x)
        ###################
        x = x.permute(0, 2, 1)
        ## Additive Attention ##
        x, attentions = self.attention_pass(x, seq_lengths)
        ###################
        #x = x.view(x.size(0), -1) # Shouldnt be necessary
        ## FNN ##
        x = self.linear_1(x)
        #########
        x = self.linear_out(x)
 #       x = x.view(-1, 1).flatten() ## Shouldnt it be x=x.flatten()
        return x, attentions

