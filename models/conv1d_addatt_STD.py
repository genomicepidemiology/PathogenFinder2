import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from collections import OrderedDict

class Attention_Methods(nn.Module):
    def __init__(self, dimensions_in, attention_dim, attention_type="Bahdanau", dropout=0.0):
        super(Attention_Methods, self).__init__()
        self.dimensions_in = dimensions_in
        self.attention_type = attention_type

        # Linear transformations for Q, K, V from the same source
        self.k_w = nn.Linear(dimensions_in, attention_dim)
        self.q_w = nn.Linear(dimensions_in, attention_dim)
        
        self.dropout = nn.Dropout(dropout)
        if attention_type == "Bahdanau":
            self.score_proj = nn.Linear(attention_dim, 1, bias=True)
        else:
            self.score_proj = None

        self.softmax_weights = nn.Softmax(dim=-1)

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

        query = self.q_w(x_in)
        key = self.k_w(x_in)
       
        weights = self.attention_pass(query=query, key=key)
        weights = self.dropout(weights)
        weights = weights.squeeze(2).unsqueeze(1)

        weights.masked_fill_(mask, -float("inf"))

        att = self.softmax_weights(weights)

        attention_result = torch.bmm(att, x_in).squeeze(1)

        return attention_result, att



class Conv1D_AddAtt_Net(nn.Module):

    def __init__(self, input_dim=1024, conv_channels=[1024*4, 1024*2,100], num_of_class=1,
                 kernel_sizes=[5,3,3], stride=1, nodes_fnn=50, attention_dim=512,
                 dropout_conv=0.2, dropout_fnn=0.3, dropout_att=0.3,
                 dropout_in=0.4, batch_norm=False, layer_norm=False, attention_type="Bahdanau"):
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
        mask = Attention_Methods.create_mask(seq_lengths=seq_lengths, dimensions_batch=x.shape)

        x = x.permute(0, 2, 1)

        ## Convolutional ##
        for layer in self.conv1d_layers:
            x = layer(x)
            x = x.masked_fill_(mask, 0)

        x = x.permute(0, 2, 1)
        ## Additive Attention ##
        x, attentions = self.attention_layer(x, mask)
        ## FNN ##
        x = self.linear_1(x)
        x = self.linear_out(x)
        return x, attentions

