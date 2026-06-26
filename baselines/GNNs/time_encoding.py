"""
time_encoding.py — shared fixed-frequency cosine time encoder (GraphMixer,
Cong et al. 2023), used by all three temporal baselines (fish_rgcn.py,
fish_tgat.py, fish_sthn.py) so "time encoding" means the same thing in every
ablation. Weights are frozen at a log-spaced frequency grid and never train;
this is deliberately not the learnable Time2Vec (Kazemi et al. 2019 / TGAT)
that fish_tgat.py and fish_sthn.py used previously.
"""

import numpy as np
import torch
import torch.nn as nn


class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
