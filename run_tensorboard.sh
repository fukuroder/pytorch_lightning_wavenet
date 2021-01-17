sudo docker run -p 6006:6006 --rm -it -v $PWD/wavenet:/wavenet pytorch_lightning_wavenet:latest tensorboard --logdir /wavenet/lightning_logs --bind_all
