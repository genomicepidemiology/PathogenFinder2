import torch
from torch import nn
from functools import partial
from .utils import LayerNorm1d, SinusoidalPositionEmbeddings, Permute


class Classifier(nn.Module):

    def __init__(self, dim, num_classes, length_information, length_dim=None):
        super(Classifier, self).__init__()

        if not length_information:
            self.length_information = False
        elif length_information == "add" or length_information == "concat1":
            self.length_information = length_information
            self.length_step = nn.Sequential(
                SinusoidalPositionEmbeddings(length_dim),
                nn.Linear(length_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                Permute([0, 2, 1]),
                )
            if length_information == "concat1":
                dim = dim*2
        else:
            self.length_information = length_information
            dim += 1
        norm_layer = partial(LayerNorm1d, eps=1e-6)
        self.norm_layer = norm_layer(dim)
        self.flatten = nn.Flatten(1)
        self.linear_out = nn.Linear(dim, num_classes)



    def forward(self, x, t=None):
        if self.length_information == "concat2":
            t = t.unsqueeze(-1)
            x = torch.concat((x, t), axis=1)
        elif self.length_information == "add":
            x = x + self.length_step(t)
        elif self.length_information == "concat1":
            x = torch.concat((x, self.length_step(t)), axis=1)
        x = self.flatten(self.norm_layer(x))
        x = self.linear_out(x)
        return x
        

