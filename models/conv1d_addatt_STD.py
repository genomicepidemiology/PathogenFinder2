import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from collections import OrderedDict

class Attention_Methods(nn.Module):
    def __init__(self, dimensions_in, attention_type="Bahdanau", dropout=0.0, init_weights=None):
        super(Attention_Methods, self).__init__()
        self.dimensions_in = dimensions_in
        self.attention_type = attention_type

        # Linear transformations for Q, K, V from the same source
        self.k_w = nn.Linear(dimensions_in, dimensions_in)
        self.q_w = nn.Linear(dimensions_in, dimensions_in)
        
        self.dropout = nn.Dropout(dropout)
        if attention_type == "Bahdanau":
            self.score_proj = nn.Linear(dimensions_in, 1, bias=True)
        else:
            self.score_proj = None

        self.attention_pass = self._get_pass() 

        self.init_weights()      

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.q_w.weight)
        self.q_w.bias.data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.k_w.weight)
        self.k_w.bias.data.fill_(0.01)

    def _get_pass(self):
        if self.attention_type == "Bahdanau":
            return self.bahdanau_pass
        else:
            raise ValueError("The type of attention {} is not avaialable".format(self.attention_type))

    def bahdanau_pass(self, query, key):
        return self.score_proj(torch.tanh(query + key))

    @staticmethod
    def create_mask(seq_lengths, dimensions_batch):
        mask = torch.arange(dimensions_batch[1], device=seq_lengths.device)[None, :] > seq_lengths[:, None]
        return mask

    def forward(self, x_in, mask):
#        mask = Attention_Methods.create_mask(seq_lengths=seq_lengths, dimensions_batch=x_in.shape)

        query = self.q_w(x_in)
        key = self.k_w(x_in)
       
        weights = self.attention_pass(query=query, key=key)
        weights = self.dropout(weights)
        weights = weights.squeeze(2).unsqueeze(1)

        weights.masked_fill_(mask, -float("inf"))

        att = torch.nn.functional.softmax(weights, dim=-1)

        attention_result = torch.bmm(att, x_in).squeeze(1)

        return attention_result, att



class Conv1D_AddAtt_Net(nn.Module):

    def __init__(self, input_dim=1024, conv_channels=[1024*4, 1024*2,100], num_of_class=1,
                 kernel_sizes=[5,3,3], stride=1, nodes_fnn=50,
                 dropout_conv=0.2, dropout_fnn=0.3, dropout_att=0.3,
                 dropout_in=0.4, batch_norm=False, layer_norm=False, attention_type="Bahdanau"):
        super(Conv1D_AddAtt_Net, self).__init__()

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

        self.layer_conv2 = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(conv_channels[0], conv_channels[1],
                                    kernel_size=kernel_sizes[1],
                                    stride=1, padding=kernel_sizes[1]//2)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))

        self.layer_conv3 = nn.Sequential(OrderedDict([
            ("conv1d", nn.Conv1d(conv_channels[1], conv_channels[2],
                                    kernel_size=kernel_sizes[2],
                                    stride=1, padding=kernel_sizes[2]//2)),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_conv))]))
        
        self.attention_layer = Attention_Methods(attention_type=attention_type,
							dimensions_in=conv_channels[2],
                                                        dropout=dropout_att)
        self.linear_1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(conv_channels[2], nodes_fnn)),
            ("activation", nn.LeakyReLU()),
            ("dropout", nn.Dropout(dropout_fnn))]))

        self.linear_out = nn.Linear(nodes_fnn, num_of_class)

        self.init_weights()

    def init_weights(self):

        torch.nn.init.kaiming_normal_(self.layer_conv1[0].weight, mode='fan_in', nonlinearity='relu')
        self.layer_conv1[0].bias.data.fill_(0.01)

        torch.nn.init.kaiming_normal_(self.layer_conv2[0].weight, mode='fan_in', nonlinearity='relu')
        self.layer_conv2[0].bias.data.fill_(0.01)

        torch.nn.init.kaiming_normal_(self.layer_conv3[0].weight, mode='fan_in', nonlinearity='relu')
        self.layer_conv3[0].bias.data.fill_(0.01)

        torch.nn.init.kaiming_normal_(self.linear_1[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        self.linear_1[0].bias.data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.linear_out.weight)
        self.linear_out.bias.data.fill_(0.01)

    @staticmethod
    def init_weights_old(module, init_weights, layer_type, nonlinearity=None):
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

    def attention_pass_old(self, x_in, seq_lengths):  # x_in.shape: [bs, seq_len, in_size]
        att_vector = self.linear_in_att(x_in) # [bs, seq_len, att_size]
        att_hid_align = self.dropout_att(self.att_act(att_vector)) # [bs, seq_len, att_size]
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

        x = self.layer_conv2(x)
        x = x.masked_fill_(mask, 0)

        x = self.layer_conv3(x)
        x = x.masked_fill_(mask, 0)

        x = x.permute(0, 2, 1)
        ## Additive Attention ##
        x, attentions = self.attention_layer(x, mask)
        ## FNN ##
        x = self.linear_1(x)
        x = self.linear_out(x)
        return x, attentions

