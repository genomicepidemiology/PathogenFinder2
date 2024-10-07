import torch
import math
from torch import nn
from functools import partial
from .utils import LayerNorm1d, Permute


class Attention_Methods(nn.Module):
    def __init__(self, dimensions_in, attention_dim, attention_type="Bahdanau", dropout=0.0, norm_layer=False):
        super(Attention_Methods, self).__init__()
        self.dimensions_in = dimensions_in
        self.attention_type = attention_type

        if norm_layer:
            norm_layer = partial(LayerNorm1d, eps=1e-6)
            self.norm_layer = norm_layer
        else:
            self.norm_layer = None


        # Linear transformations for Q, K, V from the same source
        #self.k_w = nn.Linear(dimensions_in, attention_dim, bias=False)
        #self.q_w = nn.Linear(dimensions_in, attention_dim, bias=False)
        self.k_w = nn.Sequential(Permute([0, 2, 1]),
 #                               self.norm_layer(dimensions_in),
                                nn.Conv1d(dimensions_in, attention_dim, kernel_size=1, bias=False),
                                self.norm_layer(attention_dim),
                                Permute([0, 2, 1]))
        self.q_w = nn.Sequential(Permute([0, 2, 1]),
  #                              self.norm_layer(dimensions_in),
                                nn.Conv1d(dimensions_in, attention_dim, kernel_size=1, bias=False),
                                self.norm_layer(attention_dim),
                                Permute([0, 2, 1]))
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

 #       self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.q_w.weight)
        if self.q_w.bias is not None:
            self.q_w.bias.data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.k_w.weight)
        if self.k_w.bias is not None:
            self.k_w.bias.data.fill_(0.01)

    def forward(self, x_in, mask):

        query = self.q_w(x_in)
        key = self.k_w(x_in)
 #       if self.norm_layer is None:
        projection = torch.tanh(query + key)
  #      else:
   #         projection = torch.tanh(self.norm_layer(query + key))
        weights = self.score_proj(projection)
        weights = self.dropout(weights)
        weights = weights.squeeze(2).unsqueeze(1)

        weights = weights.masked_fill(mask, -float("inf"))

        att = torch.nn.functional.softmax(weights, dim=-1)
        attention_result = torch.bmm(att, x_in).squeeze(1)

        return attention_result, att

