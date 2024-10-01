from torch import nn, Tensor
import torch
from .layers.convnextblock import CNBlock
from functools import partial
from .layers.utils import LayerNorm1d, Conv1dNormActivation, Permute, Padding1d
import models.layers.utils as utils
from typing import Union, Tuple


class ConvNeXt_Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        block_dims: list,
        num_blocks: int,
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1,
        norm: str = LayerNorm1d,
        downsample: bool = False,
        ) -> None:
        super().__init__()

        assert num_blocks == len(block_dims)

        if norm == "Layer":
            norm_layer = partial(LayerNorm1d, eps=1e-6)
        elif norm == "Batch":
            norm_layer = BatchNorm
        else:
            norm_layer = None

        self.downsample = downsample

        # Stem
        self.stem_cell = self.create_stemcell(input_dim=input_dim,
                                    output_dim=block_dims[0], norm_layer=norm_layer)
        self.features = nn.ModuleList()
        stage_block_id = 0
        for n in range(num_blocks):
            dim_block = block_dims[n]
            # Bottlenecks
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (num_blocks - 1.0)
            cnblock = self.create_block(dim=dim_block, norm_layer=norm_layer, sd_prob=sd_prob, layer_scale=layer_scale)
            self.features.append(cnblock)
            stage_block_id += 1

            if n != num_blocks-1 and dim_block != block_dims[n+1]:
                downsample_layer = self.create_downsample(dim_in=dim_block, dim_out=block_dims[n+1], norm_layer=norm_layer)
                self.features.append(downsample_layer)

        self.avgpool = self.adaptiveavgpool_mask
        self.pool_mask = nn.MaxPool1d(2, stride=2)

        self.classifier = nn.Sequential(
            norm_layer(block_dims[-1]), nn.Flatten(1), nn.Linear(block_dims[-1], num_classes)
            )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def adaptiveavgpool_mask(self, x:Tensor, lengths:Tensor) -> Tensor:
        x = torch.sum(x, axis=2)
        x = torch.div(x,lengths)
        return x[:,:,None]

    def create_downsample(self, dim_in:int, dim_out:int, norm_layer:nn.Module) -> nn.Module:
        if self.downsample:
            downsample_layer = nn.Sequential(norm_layer(dim_in),
                                    nn.Conv1d(dim_in, dim_out, kernel_size=2, stride=2, bias=False))
        else:
            downsample_layer =  nn.Sequential(norm_layer(dim_in),
                                    nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False))
        return downsample_layer

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

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Union[Tensor, None]]:
        mask = utils.create_mask(seq_lengths=lengths, dimensions_batch=x.shape)
        x = self.stem_cell(x)
        x = x.masked_fill(mask, 0)
        for layer in self.features:
            x = layer(x)
            if "block" not in layer.__dict__["_modules"].keys():
                mask = self.pool_mask(mask.float()).bool()
            x = x.masked_fill(mask, 0)
        x = self.adaptiveavgpool_mask(x, lengths)
        x = self.classifier(x)
        return x, None

