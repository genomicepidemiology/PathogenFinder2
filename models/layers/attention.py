import torch
import math
from torch import nn, Tensor
from functools import partial
from .utils import LayerNorm1d, Permute
from torchvision.ops.stochastic_depth import StochasticDepth



class Attention_Methods(nn.Module):
    def __init__(self, dimensions_in, attention_dim, residual_attention=False, attention_type="Bahdanau",
                        dropout=0.0, norm_layer=False, stochastic_depth_prob=0., layer_scale=0.):
        super(Attention_Methods, self).__init__()
        self.dimensions_in = dimensions_in
        self.attention_type = attention_type

        if norm_layer:
            norm_layer = partial(LayerNorm1d, eps=1e-6)
            self.norm_layer = norm_layer
        else:
            self.norm_layer = None

        self.k_w = nn.Sequential(
                                self.norm_layer(dimensions_in),
                                Permute([0, 2, 1]),
                                nn.Conv1d(dimensions_in, attention_dim, kernel_size=1, bias=False),
                                )
        self.q_w = nn.Sequential(
                                self.norm_layer(dimensions_in),
                                Permute([0, 2, 1]),
                                nn.Conv1d(dimensions_in, attention_dim, kernel_size=1, bias=False),
                                )
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)

        self.residual_attention = residual_attention

        if self.residual_attention:
            self.layer_scale = nn.Parameter(torch.ones(dimensions_in, 1) * layer_scale)
            self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        else:
            self.layer_scale = None
            self.stochastic_depth = None

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.q_w[-1].weight)
        if self.q_w[-1].bias is not None:
            self.q_w.bias[-1].data.fill_(0.01)

        torch.nn.init.xavier_normal_(self.k_w[-1].weight)
        if self.k_w[-1].bias is not None:
            self.k_w.bias[-1].data.fill_(0.01)

    def attention_forward(self, x_in, mask):
        x_in = x_in.permute(0, 2, 1)
        query = self.q_w(x_in)
        key = self.k_w(x_in)
        projection = torch.tanh(query + key)
        projection = projection.permute(0, 2, 1)
        weights = self.score_proj(projection)
        weights = weights.squeeze(2).unsqueeze(1)

        weights = weights.masked_fill(mask, -float("inf"))

        att = self.dropout(torch.nn.functional.softmax(weights, dim=-1))
        attention_result = torch.bmm(att, x_in).squeeze(1).unsqueeze(2)
        return attention_result, att

    def adaptiveavgpool_mask(self, x:Tensor, lengths:Tensor) -> Tensor:
        x = torch.sum(x, axis=2)
        x = torch.div(x,lengths)
        return x[:,:,None]

    def forward(self, x_in, mask, lengths=None):
        if self.residual_attention:
            att_block, att = self.attention_forward(x_in, mask)
            pool_in = self.adaptiveavgpool_mask(x_in, lengths)
            attention_result = self.layer_scale * att_block
            attention_result = self.stochastic_depth(attention_result)
            attention_result += pool_in
        else:
            attention_result, att = self.attention_forward(x_in, mask)
        return attention_result, att

