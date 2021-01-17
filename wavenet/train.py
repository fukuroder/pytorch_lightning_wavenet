import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from net import UpsampleNet
from net import WaveNet
from dataset import Dataset

class WavenetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batchsize, length, quantize, seed, process):
        super().__init__()
        self.dataset_path = dataset_path
        self.batchsize = batchsize
        self.length = length
        self.quantize = quantize
        self.seed = seed
        self.process = process

    def setup(self, stage):
        if not os.path.isdir(args.dataset):
            raise RuntimeError('Dataset directory not found: {}'.format(args.dataset))
            
        dataset = Dataset(self.dataset_path, self.length, self.quantize)
        train_dataset_len = int(len(dataset) * 0.9)
        valid_dataset_len = len(dataset) - train_dataset_len

        torch.manual_seed(self.seed)
        self.train_dataset, self.valid_dataset = random_split(
            dataset,
            [train_dataset_len, valid_dataset_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batchsize, num_workers=self.process, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.batchsize, num_workers=self.process, pin_memory=True, shuffle=False)


class WavenetLightningModule(pl.LightningModule):
    def __init__(self, n_loop, n_layer, a_channels, r_channels, s_channels, use_embed_tanh):
        super().__init__()
        self.save_hyperparameters()
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
            
    def forward(self, x):
        encoded_condition = self.encoder(x[1])
        y = self.decoder(x[0].float(), encoded_condition)
        return y

    def training_step(self, batch, batch_nb):
        pred = F.log_softmax(self(batch[:-1]), dim=1)
        pred_idx = pred.argmax(1)
        loss = F.nll_loss(pred, batch[-1].long(), reduction='none').mean(dim=1)
        acc = torch.mean((pred_idx == batch[-1]).float(), dim=1)
        return {'loss': loss, 'acc': acc}
        
    def training_step_end(self, outputs):
        return {'loss': outputs["loss"].mean(), 'acc': outputs["acc"].mean()}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log_dict({"train/loss": loss, "train/acc": acc})

    def validation_step(self, batch, batch_nb):
        pred = F.log_softmax(self(batch[:-1]), dim=1)
        pred_idx = pred.argmax(1)
        loss = F.nll_loss(pred, batch[-1].long(), reduction='none').mean(dim=1)
        acc = torch.mean((pred_idx == batch[-1]).float(), dim=1)
        return {'loss': loss, 'acc': acc}
        
    def validation_step_end(self, outputs):
        return {'loss': outputs["loss"].mean(), 'acc': outputs["acc"].mean()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log_dict({"val/loss": loss, "val/acc": acc})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch-lightning-wavenet')
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Numer of audio clips in each mini-batch')
    parser.add_argument('--length', '-l', type=int, default=7680,
                        help='Number of samples in each audio clip')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-i', default='./VCTK-Corpus',
                        help='Directory of dataset')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_loop', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--n_layer', type=int, default=10,
                        help='Number of layers in each residual block')
    parser.add_argument('--a_channels', type=int, default=256,
                        help='Number of channels in the output layers')
    parser.add_argument('--r_channels', type=int, default=64,
                        help='Number of channels in residual layers and embedding')
    parser.add_argument('--s_channels', type=int, default=256,
                        help='Number of channels in the skip layers')
    parser.add_argument('--use_embed_tanh', type=bool, default=True,
                        help='Use tanh after an initial 2x1 convolution')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to split dataset into train and test')
    parser.add_argument('--process', type=int, default=2,
                        help='Number of parallel processes')
    parser.add_argument('--gpus', '-g', type=int, default=-1,
                        help='Number of GPUs')
    args = parser.parse_args()
   
    data_module = WavenetDataModule(
        dataset_path=args.dataset,
        batchsize=args.batchsize,
        length=args.length,
        quantize=args.a_channels,
        seed=args.seed,
        process=args.process)

    if args.resume:
        model = WavenetLightningModule.load_from_checkpoint(args.resume)
    else:
        model = WavenetLightningModule(
            args.n_loop,
            args.n_layer,
            args.a_channels,
            args.r_channels,
            args.s_channels,
            args.use_embed_tanh)

    callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min', save_last=True)
    trainer = pl.Trainer(
        gpus=args.gpus,
        accelerator='dp' if args.gpus !=0 else None,
        max_epochs=args.epoch,
        callbacks=[callback],
        resume_from_checkpoint=args.resume if args.resume else None,
        benchmark = True)    
    trainer.fit(model, data_module) 
