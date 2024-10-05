#!/bin/bash

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/protodpmm_net.yml \
    configs/pipelines/train/train_protodpmm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --seed 0
