import torch
import torch.nn.functional as F
from torch import nn

from modules import ResidualNet


class UpsampleNet(nn.ModuleList):
    def __init__(self, out_layers=40, r_channels=64, channels=[128, 128, 128], upscale_factors=[16, 16]):
        super().__init__()
        self.n_deconvolutions = 0
        for in_channel, out_channel, factor in zip(channels[:-1], channels[1:], upscale_factors):
            self.append(nn.ConvTranspose1d(
                in_channel, out_channel, factor, stride=factor, padding=0))
            self.n_deconvolutions += 1
        for _ in range(out_layers):
            self.append(nn.Conv1d(channels[-1], 2 * r_channels, 1))

    def forward(self, x):
        conditions = []
        for i, layer in enumerate(self):
            if i < self.n_deconvolutions:
                x = F.relu(layer(x))
            else:
                conditions.append(layer(x))
        return torch.stack(conditions)


class WaveNet(nn.Module):
    def __init__(self, n_loop=4, n_layer=10, a_channels=256, r_channels=64, s_channels=256, use_embed_tanh=True):
        super().__init__()
        self.embed = nn.Conv1d(
            a_channels, r_channels, 2, padding=1, bias=False)
        self.resnet = ResidualNet(
            n_loop, n_layer, 2, r_channels, 2 * r_channels, s_channels)
        self.proj1 = nn.Conv1d(
            s_channels, s_channels, 1, bias=False)
        self.proj2 = nn.Conv1d(
            s_channels, a_channels, 1, bias=False)
        self.a_channels = a_channels
        self.s_channels = s_channels
        self.use_embed_tanh = use_embed_tanh

    def forward(self, x, condition):
        length = x.size(2)
        x = self.embed(x)
        x = x[:, :, :length]  # crop
        if self.use_embed_tanh:
            x = torch.tanh(x)
        z = F.relu(self.resnet(x, condition))
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        return y

    def initialize(self, n):
        self.resnet.initialize(n)

        self.embed.padding = 0
        self.embed_queue = torch.zeros(
            (n, self.a_channels, 2), dtype=self.embed.weight.dtype)

        self.proj1_queue = torch.zeros(
            (n, self.s_channels, 1), dtype=self.proj1.weight.dtype)

        self.proj2_queue = torch.zeros(
            (n, self.s_channels, 1), dtype=self.proj2.weight.dtype)

    def generate(self, x, condition):
        self.embed_queue = torch.cat((self.embed_queue[:, :, 1:], x), dim=2)
        x = self.embed(self.embed_queue)
        if self.use_embed_tanh:
            x = torch.tanh(x)
        x = F.relu(self.resnet.generate(x, condition))
        self.proj1_queue = torch.cat((self.proj1_queue[:, :, 1:], x), dim=2)
        x = F.relu(self.proj1(self.proj1_queue))
        self.proj2_queue = torch.cat((self.proj2_queue[:, :, 1:], x), dim=2)
        x = self.proj2(self.proj2_queue)
        return x
