import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from .conv1d_addatt_STD import Attention_Methods

class Conv1D_Net(nn.Module):

    def __init__(self, input_dim=1024, num_of_class=1,
                 kernel_sizes=[5], stride=1, conv_channels=[100],
                 dropout_conv=0.2, dropout_in=0.4, batch_norm=False, layer_norm=False):
        super(Conv1D_Net, self).__init__()

        if batch_norm:
            self.in_layer = nn.Sequential(OrderedDict([
                ("batchnorm", nn.BatchNorm2d(input_dim)),
                ("drop_in", nn.Dropout1d(dropout_in))
            ]))
        else:
            self.in_layer = nn.Sequential(OrderedDict([
                ("drop_in", nn.Dropout1d(dropout_in))
            ]))

        self.layer_conv1 = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(input_dim, conv_channels[0],
                        kernel_size=kernel_sizes[0],
                        stride=1, padding=kernel_sizes[0]//2)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))


        self.linear_out = nn.Linear(conv_channels[0], num_of_class)

        self.init_weights()

    def init_weights(self):

        torch.nn.init.kaiming_normal_(self.layer_conv1[0].weight, mode='fan_in', nonlinearity='relu')
        self.layer_conv1[0].bias.data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.linear_out.weight)
        self.linear_out.bias.data.fill_(0.01)

    def attention_pass_old(self, x_in, seq_lengths):  # x_in.shape: [bs, seq_len, in_size]
        att_vector = self.linear_in_att(x_in) # [bs, seq_len, att_size]
        att_hid_align = self.dropout_att(self.att_act(att_vector)) # [bs, seq_len, att_size]
        att_score = self.linear_att(att_hid_align).squeeze(2) # [bs, seq_len]
        mask = Conv1D_AddAtt_Net.length_to_negative_mask(seq_lengths)
        alpha = F.softmax(att_score + mask, dim=1) # [bs, seq_len]
        att = alpha.unsqueeze(2) # [bs, seq_len, 1]
        return torch.sum(x_in * att, dim=1), alpha # [bs, in_size]

    def masked_AvgPool(self, x, seq_lengths):
        x = torch.sum(x, 2)
        x = torch.divide(x, seq_lengths)
        return x

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

    def attention_pass(self, x_in, seq_lengths):
        q = nn.Linear(x_in, x_in, bias=False)
        k = nn.Linear(x_in, x_in, bias=False)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
  
        mask = (mask == 0).view(mask_reshape).expand_as(att)
        att.masked_fill_(mask, -float("inf"))
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        return torch.sum(x_in * att, dim=1), att

    def forward(self, x, seq_lengths):
        x = self.in_layer(x)
        mask = Attention_Methods.create_mask(seq_lengths=seq_lengths, dimensions_batch=x.shape)

        x = x.permute(0, 2, 1)
        ## Convolutional ##
        x = self.layer_conv1(x)
        x = x.masked_fill_(mask, 0)

        x = self.masked_AvgPool(x, seq_lengths)
        x = torch.squeeze(x, 1)
        x = self.linear_out(x)
        return x, None

