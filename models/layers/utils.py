import torch


def create_mask(seq_lengths, dimensions_batch):
    mask = torch.arange(dimensions_batch[1], device=seq_lengths.device)[None, :] > seq_lengths[:, None]
    return mask
