import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

import models.layers.utils as utils
from models.layers.attention import Attention_Methods


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385

    def __init__(self, in_dim, out_dim, factor_bottle=2, kernel=3, groups=1):
        super(Bottleneck, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        middle_dim = int(in_dim/factor_bottle)

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(in_dim, middle_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(middle_dim)
        self.conv2 = nn.Conv1d(middle_dim, middle_dim, kernel_size=kernel, padding=kernel//2, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(middle_dim)
        self.conv3 = nn.Conv1d(middle_dim, out_dim, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=False)

        if in_dim != out_dim:
            self.downsample = nn.Sequential(
					nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=False),
					nn.BatchNorm1d(self.out_dim))
        else:
            self.downsample = None

    def _init_weights(self):

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
#        self.conv1.bias.data.fill_(0.01)        
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
#        self.conv1.bias.data.fill_(0.01)
        torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
#        self.conv1.bias.data.fill_(0.01)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.in_dim != self.out_dim:
            identity = self.downsample(x)

        out = torch.add(out,  identity)
        out = F.relu(out)

        return out

class DenseNet_AddAtt_Net(nn.Module):

    def __init__(self, input_dim, num_of_class, num_blocks, kernel_sizes, nodes_fnn, conv_channels,
                        dropout_in, factor_block, dropout_fnn, dropout_att, attention_dim, batch_norm,
                        layer_norm=False, attention_type="Bahdanau", stride=1):
        super(DenseNet_AddAtt_Net, self).__init__()
        assert len(kernel_sizes) == len(conv_channels)
        assert num_blocks == len(kernel_sizes)

        if batch_norm:
            dropout_fnn = 0

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

        self.attention_layer = Attention_Methods(attention_type=attention_type,
                                                        dimensions_in=input_dim,
                                                        attention_dim=attention_dim,
                                                        dropout=dropout_att)
        self.linear_1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(input_dim, nodes_fnn)),
            ("batch_norm", nn.BatchNorm1d(nodes_fnn)),
            ("activation", nn.LeakyReLU()),
            ("dropout", nn.Dropout(dropout_fnn))]))

        if not batch_norm:
            del self.linear_1[1]

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

