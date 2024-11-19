from torch import nn, Tensor
import torch
from functools import partial
from torchvision.ops.stochastic_depth import StochasticDepth 
from torchvision.ops import stochastic_depth
from .utils import Permute, Padding1d

class CNBlock(nn.Module):
    multiply_factor = 4

    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        kernel_size: int = 7,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False),
            Permute([0,2,1]),
            norm_layer(dim),
            Permute([0, 2, 1]),
            nn.Conv1d(dim, dim*CNBlock.multiply_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(dim*CNBlock.multiply_factor, dim, kernel_size=1, stride=1, padding=0, bias=False),
            )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        block = self.block(input)
        result = self.layer_scale * block
        result = self.stochastic_depth(result)
        result += input
        return result
