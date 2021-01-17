import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, filter_size, dilation, residual_channels, dilated_channels, skip_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            residual_channels, dilated_channels,
            kernel_size=filter_size,
            padding=dilation * (filter_size - 1), dilation=dilation)
        self.res = nn.Conv1d(
            dilated_channels // 2, residual_channels, 1)
        self.skip = nn.Conv1d(
            dilated_channels // 2, skip_channels, 1)

        self.filter_size = filter_size
        self.dilation = dilation
        self.residual_channels = residual_channels

    def forward(self, x, condition):
        length = x.shape[2]
        h = self.conv(x)
        h = h[:, :, :length]  # crop
        h += condition
        tanh_z, sig_z = torch.split(h, h.size(1)//2, dim=1)
        z = torch.tanh(tanh_z) * torch.sigmoid(sig_z)
        if x.shape[2] == z.shape[2]:
            residual = self.res(z) + x
        else:
            residual = self.res(z) + x[:, :, -1:]  # crop
        skip_connection = self.skip(z)
        return residual, skip_connection

    def initialize(self, n):
        self.queue = torch.zeros((
            n, self.residual_channels,
            self.dilation * (self.filter_size - 1) + 1),
            dtype=self.conv.weight.dtype)
        self.conv.padding = 0

    def pop(self, condition):
        return self(self.queue, condition)

    def push(self, x):
        self.queue = torch.cat((self.queue[:, :, 1:], x), dim=2)


class ResidualNet(nn.ModuleList):
    def __init__(self, n_loop=4, n_layer=10, filter_size=2, residual_channels=64, dilated_channels=128, skip_channels=256):
        super().__init__()
        dilations = [2 ** i for i in range(n_layer)] * n_loop
        for dilation in dilations:
            self.append(ResidualBlock(filter_size, dilation, residual_channels, dilated_channels, skip_channels))

    def forward(self, x, conditions):
        for i, (func, cond) in enumerate(zip(self, conditions)):
            x, skip = func(x, cond)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections

    def initialize(self, n):
        for block in self:
            block.initialize(n)

    def generate(self, x, conditions):
        for i, (func, cond) in enumerate(zip(self, conditions)):
            func.push(x)
            x, skip = func.pop(cond)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections
