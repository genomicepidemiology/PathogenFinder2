from torch import nn, Tensor
import torch
from dl.models.layers.convnextblock import CNBlock
from functools import partial
from dl.models.layers.utils import LayerNorm1d, Conv1dNormActivation, Permute
from typing import Union, Tuple
from torchvision.ops.stochastic_depth import StochasticDepth
from dl.models.layers.attention import Attention_Methods
from dl.models.layers.classifier import Classifier
import dl.models.layers.utils as utils



class ConvNext_AddAtt_Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        block_dims: int,
        num_blocks: int,
        attention_dim: int,
        dropout_att: float,
        fnn_dim: int = 0,
        stem_cell: bool = True,
        sequence_dropout: float = 0.3,
        length_information = False,
        length_dim = None,
        attention_type: str = "Bahdanau",
        residual_attention: bool = False,
        stochastic_depth_prob: float = 0.0,
        stochastic_depth_att: bool = False,
        layer_scale: float = 1e-6,
        num_classes: int = 1,
        norm: str = "Layer",
        attention_norm: bool = False,
        ) -> None:
        super().__init__()

        block_dims_lst = ConvNext_AddAtt_Net.get_blocks_lst(num_blocks=num_blocks, blocks_dim=block_dims)
        self.input_dropout = nn.Dropout1d(sequence_dropout)
        self.num_classes = num_classes

        if norm == "Layer":
            norm_layer = partial(LayerNorm1d, eps=1e-6)
        elif norm == "Batch":
            norm_layer = BatchNorm
        else:
            norm_layer = None

        self.stage_block_id = 1
        self.stochastic_depth_prob = stochastic_depth_prob
        self.num_blocks = 1 + len(block_dims_lst) + int(stochastic_depth_att)
      
        # Stem
        if stem_cell:
            self.stem_cell = self.create_stemcell(input_dim=input_dim,
                                    output_dim=block_dims_lst[0], norm_layer=norm_layer)
        else:
            self.stem_cell = False

        self.features = nn.ModuleList()
        for n in range(len(block_dims_lst)):
            dim_block = block_dims_lst[n]
            # Bottlenecks
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = self.get_sd_prob()
            cnblock = self.create_block(dim=dim_block, norm_layer=norm_layer, sd_prob=sd_prob, layer_scale=layer_scale)
            self.features.append(cnblock)
            if n != len(block_dims_lst)-1 and dim_block != block_dims_lst[n+1]:
                downsample = self.create_downsample(dim_in=dim_block, dim_out=block_dims_lst[n+1], norm_layer=norm_layer)
                self.features.append(downsample)

        self.apply(self._init_weights)

        att_sd = self.get_sd_prob()
        self.residual_attention = residual_attention
        self.attention_layer = Attention_Methods(attention_type=attention_type,
                                                        dimensions_in=block_dims_lst[-1],
                                                        attention_dim=attention_dim,
                                                        norm_layer=attention_norm,
                                                        dropout=dropout_att,
                                                        residual_attention=residual_attention,
                                                        stochastic_depth_prob=att_sd, layer_scale=layer_scale)

        if self.residual_attention: 
            self.stochastic_depth_att = False
        else:
            if stochastic_depth_att:
                self.stochastic_depth_att = StochasticDepth(att_sd, "row")
            else:
                self.stochastic_depth_att = False
        if fnn_dim != 0:
            self.fnn_out = nn.Sequential(norm_layer(block_dims_lst[-1]), nn.Linear(block_dims_lst[-1],fnn_dim), nn.ReLU())
            nn.init.kaiming_normal_(self.fnn_out[-2].weight)
            self.fnn_out[-2].data.fill(0.01)
            inclass_dim = fnn_dim
        else:
            self.fnn_out = None
            inclass_dim = block_dims_lst[-1]

        self.classifier = Classifier(inclass_dim, num_classes, length_information, length_dim)

        nn.init.xavier_normal_(self.classifier.linear_out.weight)
        self.classifier.linear_out.bias.data.fill_(0.01)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



    @staticmethod
    def get_blocks_lst(num_blocks, blocks_dim):

        blocks_lst = [blocks_dim]*(num_blocks-1)
        blocks_lst.append(int(blocks_dim/2))
        return blocks_lst


    def get_sd_prob(self):
        sd_prob = self.stochastic_depth_prob * self.stage_block_id / (self.num_blocks - 1.)
        self.stage_block_id += 1
        return sd_prob

    def create_downsample(self, dim_in:int, dim_out:int, norm_layer:nn.Module) -> nn.Module:
        return nn.Sequential(
                            Permute([0,2,1]),
                            norm_layer(dim_in),
                            Permute([0,2,1]),
                            nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
                                )

    def create_block(self, dim:int, norm_layer:nn.Module, sd_prob:float, layer_scale:float) -> nn.Module:
        return CNBlock(dim=dim, layer_scale=layer_scale, stochastic_depth_prob=sd_prob,
                        kernel_size=7, norm_layer=norm_layer)

    def create_stemcell(self, input_dim:int, output_dim:int, norm_layer: nn.Module) -> nn.Module:
        stemcell = nn.Sequential(
                    Permute([0, 2, 1]),
                    nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False),
                    Permute([0, 2, 1]),
                    norm_layer(output_dim),
                    Permute([0, 2, 1])
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
        x = x.masked_fill(mask, 0)
        for layer in self.features:
            x = layer(x)
            x = x.masked_fill(mask, 0)
        x, attentions = self.attention_layer(x, mask, lengths)
        if self.stochastic_depth_att and not self.residual_attention:
            x = self.stochastic_depth_att(x)
        print("Presqueeze", x.shape)
        x = torch.squeeze(x,dim=-1)
        if self.fnn_out is not None:
            x = self.fnn_out(x)
        x = self.classifier(x, lengths)
        return x, attentions

    @staticmethod
    def set_model_params(model, parameters):
        return model(input_dim=parameters["Input Dimensions"],
                        block_dims=parameters["Network Structure"]["Block Dimensions"],
                        num_blocks=parameters["Network Structure"]["Num Blocks"],
                        attention_dim=parameters["Network Structure"]["Attention Dimensions"],
                        dropout_att=parameters["Attention Dropout"],
                        fnn_dim=parameters["Network Structure"]["FNN Dimensions"],
                        stem_cell=parameters["Network Structure"]["Stem Cell"],
                        sequence_dropout=parameters["Sequence Dropout"],
                        length_information=parameters["Network Structure"]["Length Information"],
                        length_dim=parameters["Network Structure"]["Length Dimensions"],
                        residual_attention=parameters["Network Structure"]["Residual Attention"],
                        stochastic_depth_prob=parameters["Stochastic Depth Prob"],
                        stochastic_depth_att=parameters["Stochastic Depth Prob Att"],
                        layer_scale=parameters["Norm Scale"],
                        num_classes=parameters["Out Dimensions"],
                        norm=parameters["Norm Type"],
                        attention_norm=parameters["Attention Norm"])

