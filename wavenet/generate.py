import argparse

import torch
import torch.nn.functional as F
import librosa
import numpy
import tqdm
import soundfile as sf

from net import UpsampleNet
from net import WaveNet
from dataset import MuLaw
from dataset import Preprocess

import pytorch_lightning as pl

class WavenetLightningModule(pl.LightningModule):
    def __init__(self, n_loop, n_layer, a_channels, r_channels, s_channels, use_embed_tanh):
        super().__init__()
        self.a_channels = a_channels
        self.encoder = UpsampleNet(
            n_loop*n_layer,
            r_channels)
        self.decoder = WaveNet(
            n_loop,
            n_layer,
            a_channels,
            r_channels,
            s_channels,
            use_embed_tanh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input file')
    parser.add_argument('--output', '-o', default='result.wav', help='output file')
    parser.add_argument('--model', '-m', required=True, help='snapshot of trained model')
    args = parser.parse_args()

    # Load trained parameters
    model = WavenetLightningModule.load_from_checkpoint(args.model)
    model.eval()

    # Preprocess
    _, conditions, _ = Preprocess(
        sr=16000, n_fft=1024, hop_length=256, n_mels=128, top_db=20,
        length=None, quantize=model.a_channels)(args.input)
    conditions = torch.Tensor(conditions).unsqueeze(0)

    # Non-autoregressive generate
    encoded_conditions = model.encoder(conditions)

    # Autoregressive generate
    model.decoder.initialize(1)
    x = torch.zeros((1, model.a_channels, 1), dtype=torch.float32)
    output = numpy.zeros(encoded_conditions.size(3), dtype=numpy.float32)
    for i in tqdm.tqdm(range(len(output))):
        with torch.no_grad():
            out = model.decoder.generate(x, encoded_conditions[:, :, :, i:i + 1])
        p = F.softmax(out, dim=1).detach().numpy()[0, :, 0]
        value = numpy.random.choice(model.a_channels, size=1, p=p)[0]
        x = torch.zeros_like(x)
        x[:, value, :] = 1
        output[i] = value

    # Save
    wave = MuLaw(model.a_channels).itransform(output)
    sf.write(args.output, wave, 16000)
