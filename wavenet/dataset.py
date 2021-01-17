import torch
import random
import pathlib
import librosa
import numpy

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, length, quantize):
        super().__init__()
        self.paths = sorted([str(path) for path in pathlib.Path(dataset_path).glob('wav48/*/*.wav')])
        self.transform = Preprocess(
            sr=16000, n_fft=1024, hop_length=256, n_mels=128, top_db=20,
            length=length, quantize=quantize)

    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, i):
        return self.transform(self.paths[i])
        
class MuLaw(object):
    def __init__(self, mu=256, int_type=numpy.int32, float_type=numpy.float32):
        self.mu = mu
        self.int_type = int_type
        self.float_type = float_type

    def transform(self, x):
        x = x.astype(self.float_type)
        y = numpy.sign(x) * numpy.log(1 + self.mu * numpy.abs(x)) / \
            numpy.log(1 + self.mu)
        y = numpy.digitize(y, 2 * numpy.arange(self.mu) / self.mu - 1) - 1
        return y.astype(self.int_type)

    def itransform(self, y):
        y = y.astype(self.float_type)
        y = 2 * y / self.mu - 1
        x = numpy.sign(y) / self.mu * ((1 + self.mu) ** numpy.abs(y) - 1)
        return x.astype(self.float_type)

class Preprocess(object):
    def __init__(self, sr, n_fft, hop_length, n_mels, top_db,
                 length, quantize, dtype=numpy.float32):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        self.mu_law = MuLaw(quantize)
        self.quantize = quantize
        if length is None:
            self.length = None
        else:
            self.length = length + 1
        self.dtype = dtype

    def __call__(self, path):
        # load data with trimming and normalizing
        raw, _ = librosa.load(path, self.sr, res_type='kaiser_fast')
        raw, _ = librosa.effects.trim(raw, self.top_db)
        raw /= numpy.abs(raw).max()
        raw = raw.astype(numpy.float32)

        # mu-law transform
        quantized = self.mu_law.transform(raw)

        # padding/triming
        if self.length is not None:
            if len(raw) <= self.length:
                # padding
                pad = self.length - len(raw)
                raw = numpy.concatenate(
                    (raw, numpy.zeros(pad, dtype=numpy.float32)))
                quantized = numpy.concatenate(
                    (quantized, self.quantize // 2 * numpy.ones(pad)))
                quantized = quantized.astype(numpy.int32)
            else:
                # triming
                start = random.randint(0, len(raw) - self.length - 1)
                raw = raw[start:start + self.length]
                quantized = quantized[start:start + self.length]

        # calculate mel-spectrogram
        spectrogram = librosa.feature.melspectrogram(
            raw, self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels)
        spectrogram = librosa.power_to_db(
            spectrogram, ref=numpy.max)

        # normalize mel spectrogram into [-1, 1]
        spectrogram += 40
        spectrogram /= 40
        if self.length is not None:
            spectrogram = spectrogram[:, :self.length // self.hop_length]
        spectrogram = spectrogram.astype(self.dtype)

        # expand dimensions
        one_hot = numpy.identity(
            self.quantize, dtype=numpy.int32)[quantized].T

        return one_hot[:, :-1], spectrogram, quantized[1:]
