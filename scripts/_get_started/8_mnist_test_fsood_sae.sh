#!/bin/bash
# sh scripts/_get_started/8_mnist_test_fsood_nda.sh

GPU=1
CPU=1
node=69
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/benchmark_digits.yml \
configs/networks/lenet.yml \
configs/pipelines/train/sae.yml \
--dataset.image_size 28 \
--network.name lenet \
--num_workers 8