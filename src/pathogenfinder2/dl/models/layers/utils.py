import torch
import math
from torch.nn import functional as F
from typing import List
from torch import nn, Tensor

def masked_avg_pool(x, seq_lengths):
    """Sum-pool x over the sequence dimension then divide by sequence length."""
    x = torch.sum(x, 2)
    x = torch.divide(x, seq_lengths)
    return x

def create_mask(seq_lengths, dimensions_batch):
    mask = torch.arange(dimensions_batch[1], device=seq_lengths.device)[None, :] > seq_lengths[:, None]
    return mask

class Padding1d(nn.Module):
    def __init__(self, mask_value = 0):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, x, mask):
 #       x = x.permute(0, 2,1)
        x = x.masked_fill(mask, self.mask_value)
#        x = x.permute(0, 2,1)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LayerNorm1d(nn.LayerNorm):
    def forward(self, x:Tensor) -> Tensor:
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)
