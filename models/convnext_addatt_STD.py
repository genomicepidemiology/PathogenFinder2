from torch import nn, Tensor
import torch
from .layers.convnextblock import CNBlock
from functools import partial
from .layers.utils import LayerNorm1d, Conv1dNormActivation, Permute
from typing import Union, Tuple
from torchvision.ops.stochastic_depth import StochasticDepth
from models.layers.attention import Attention_Methods
from models.layers.classifier import Classifier
import models.layers.utils as utils



class ConvNet_AddAtt_Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        block_dims: list,
        num_blocks: int,
        attention_dim: int,
        dropout_att: float,
        sequence_dropout: float = 0.3,
        length_information = False,
        length_dim = None,
        attention_type: str = "Bahdanau",
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1,
        norm: str = "Layer",
        attention_norm: bool = False,
        ) -> None:
        super().__init__()

#        assert num_blocks == len(block_dims)

        self.input_dropout = nn.Dropout1d(sequence_dropout)

        if norm == "Layer":
            norm_layer = partial(LayerNorm1d, eps=1e-6)
        elif norm == "Batch":
            norm_layer = BatchNorm
        else:
            norm_layer = None

        self.stage_block_id = 0
        self.stochastic_depth_prob = stochastic_depth_prob
        self.num_blocks = num_blocks

        # Stem
        self.stem_cell = self.create_stemcell(input_dim=input_dim,
                                    output_dim=block_dims[0], norm_layer=norm_layer)
        sdvalue_stem_cell = self.get_sd_prob()
        self.sd_stem_cell = StochasticDepth(sdvalue_stem_cell, "row")

        self.features = nn.ModuleList()

        for n in range(len(block_dims)):
            dim_block = block_dims[n]
            # Bottlenecks
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = self.get_sd_prob()
            cnblock = self.create_block(dim=dim_block, norm_layer=norm_layer, sd_prob=sd_prob, layer_scale=layer_scale)
            self.features.append(cnblock)

            if n != len(block_dims)-1 and dim_block != block_dims[n+1]:
                self.features.append(self.create_downsample(dim_in=dim_block, dim_out=block_dims[n+1], norm_layer=norm_layer))

        self.att_norm = norm_layer(block_dims[-1])
        self.attention_layer = Attention_Methods(attention_type=attention_type,
                                                        dimensions_in=block_dims[-1],
                                                        attention_dim=attention_dim,
                                                        norm_layer=attention_norm,
                                                        dropout=dropout_att)
        att_stochastic_depth = self.get_sd_prob()
        self.stochastic_depth_att = StochasticDepth(att_stochastic_depth, "row")

        self.classifier = Classifier(block_dims[-1], num_classes, length_information, length_dim)

#        self.classifier = nn.Sequential(
 #           norm_layer(block_dims[-1]), nn.Flatten(1), nn.Linear(block_dims[-1], num_classes)
  #          )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth_prob * self.stage_block_id / (self.num_blocks - 1.)
        self.stage_block_id += 1
        return sd_prob

    def create_downsample(self, dim_in:int, dim_out:int, norm_layer:nn.Module) -> nn.Module:
        return nn.Sequential(norm_layer(dim_in),
                                nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
                                )

    def create_block(self, dim:int, norm_layer:nn.Module, sd_prob:float, layer_scale:float) -> nn.Module:
        return CNBlock(dim=dim, layer_scale=layer_scale, stochastic_depth_prob=sd_prob,
                        kernel_size=7, norm_layer=norm_layer)

    def create_stemcell(self, input_dim:int, output_dim:int, norm_layer: nn.Module) -> nn.Module:
        stemcell = nn.Sequential(
                    Permute([0, 2, 1]),
                    nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False),
                    norm_layer(output_dim),
                    )
        return stemcell

    def create_stemcell_fancy(self, input_dim: int, output_dim: int, norm_layer: nn.Module)->nn.Module:
        stem_cell = Conv1dNormActivation(
                        input_dim,
                        output_dim,
                        kernel_size=1,
                        stride=1,
                        padding=1//2,
                        norm_layer=norm_layer,
                        activation_layer=None,
                        bias=False,
                        )
        return stem_cell

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Union[Tensor, None]]:
        mask = utils.create_mask(seq_lengths=lengths, dimensions_batch=x.shape)
        x = self.input_dropout(x)
        x = self.stem_cell(x)
        x = self.sd_stem_cell(x)
        x = x.masked_fill(mask, 0)
        for layer in self.features:
            x = layer(x)
            x = x.masked_fill(mask, 0)
 #       x = self.att_norm(x)
        x = torch.permute(x, (0, 2, 1))
        x, attentions = self.attention_layer(x, mask)
        x = self.stochastic_depth_att(x)
        x = torch.unsqueeze(x, -1)
        x = self.classifier(x, lengths)
        return x, attentions

