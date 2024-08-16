import torch
import math
from torch import nn


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

    def forward(self, x_in, mask):

        query = self.q_w(x_in)
        key = self.k_w(x_in)

        weights = self.attention_pass(query=query, key=key)
        weights = self.dropout(weights)
        weights = weights.squeeze(2).unsqueeze(1)

        weights.masked_fill(mask, -float("inf"))

        att = torch.nn.functional.softmax(weights, dim=-1)

        attention_result = torch.bmm(att, x_in).squeeze(1)

        return attention_result, att

