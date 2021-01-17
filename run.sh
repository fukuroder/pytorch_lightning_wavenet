sudo docker run --shm-size=512m --gpus all --rm -v $PWD/wavenet:/wavenet -w /wavenet -it pytorch_lightning_wavenet:latest bash
