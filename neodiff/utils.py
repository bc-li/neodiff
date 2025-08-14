import math

import torch
from torch import nn

_activations = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

def build_ffn(in_dim, ffn_dim, out_dim, activation_fn="relu", drop_out=None):
    ffn =  nn.Sequential(
        nn.Linear(in_dim, ffn_dim),
        _activations[activation_fn](),
        nn.Linear(ffn_dim, out_dim),
    )

    if drop_out is not None:
        ffn.append(nn.Dropout(drop_out))

    return ffn


def timestep_embedding(timesteps, dim, max_period=0.001):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[..., None].float() * freqs[None, None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
