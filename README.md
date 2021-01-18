# pytorch_lightning_wavenet
A PytorchLightning implementation of mel-spectrogram vocoder using WaveNet. 
Created with reference to [Chainer implementation](https://github.com/chainer/chainer/tree/master/examples/wavenet).

## Usage
1. Bulid Docker image.
    - `sudo docker build -t pytorch_lightning_wavenet .`
2. Run Docker container.
    - `sudo docker run --shm-size=512m --gpus all --rm -v $PWD/wavenet:/wavenet -w /wavenet -it pytorch_lightning_wavenet:latest bash`
3. Download dataset.
    - `wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz`
    - `tar -xf VCTK-Corpus.tar.gz`
5. Start training.
    - `python train.py --dataset <directory of dataset e.g. ./VCTK-Corpus/>`
6. Generate audio with trained model.
    - `python generate.py -i <input file> -m <trained model e.g. ./lightning_logs/version_0/checkpoints/last.ckpt>`
