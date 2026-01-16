#!/usr/bin/env bash
source activate sam
cd /home/cao/code/tensor-bar

seed=44

CUDA_VISIBLE_DEVICES=0 \
python cifar_trainer.py \
    --wandb_project "cifar10-label-noise" \
    --num_epochs 200 \
    --batch_size 128 \
    --cifar 10 \
    --noise_rate 0.2 \
    --seed $seed \
    --resnet_depth 32 \
    --method gbar2 \
    --gbar_alpha 0.1 \
    --gbar_alpha_scheduler cosine \
    --num_workers 5 \
    --use_tnn 1 \
    --weight_model_class TensorRing \
    --rank 15 &

CUDA_VISIBLE_DEVICES=1 \
python cifar_trainer.py \
    --wandb_project "cifar10-label-noise" \
    --num_epochs 200 \
    --batch_size 128 \
    --cifar 10 \
    --noise_rate 0.4 \
    --seed $seed \
    --resnet_depth 32 \
    --method gbar2 \
    --gbar_alpha 0.1 \
    --gbar_alpha_scheduler cosine \
    --num_workers 5 \
    --use_tnn 1 \
    --weight_model_class TensorRing \
    --rank 15 &

CUDA_VISIBLE_DEVICES=2 \
python cifar_trainer.py \
    --wandb_project "cifar10-label-noise" \
    --num_epochs 200 \
    --batch_size 128 \
    --cifar 10 \
    --noise_rate 0.6 \
    --seed $seed \
    --resnet_depth 32 \
    --method gbar2 \
    --gbar_alpha 0.1 \
    --gbar_alpha_scheduler cosine \
    --num_workers 5 \
    --use_tnn 1 \
    --weight_model_class TensorRing \
    --rank 15 &

CUDA_VISIBLE_DEVICES=3 \
python cifar_trainer.py \
    --wandb_project "cifar10-label-noise" \
    --num_epochs 200 \
    --batch_size 128 \
    --cifar 10 \
    --noise_rate 0.8 \
    --seed $seed \
    --resnet_depth 32 \
    --method gbar2 \
    --gbar_alpha 0.1 \
    --gbar_alpha_scheduler cosine \
    --num_workers 5 \
    --use_tnn 1 \
    --weight_model_class TensorRing \
    --rank 15

