#!/usr/bin/env bash
source activate sam
cd /home/cao/code/tensor-bar

weight_model_class=$1 # CP or TensorRing
method=$2
use_tnn=$3


if [ -z "$weight_model_class" ]; then
    echo "Usage: $0 <weight_model_class>"
    exit 1
fi

if [ "$weight_model_class" = "CP" ] ; then
    ranks=(10 20 50 100 200) # CP
fi
if [ "$weight_model_class" = "TensorRing" ] ; then
    ranks=(5 7 12 17 23) # TensorRing
fi
if [ "$weight_model_class" = "Tucker" ] ; then
    ranks=(6 9 13 21 24) # Tucker
fi
if [ "$use_tnn" = "0" ] ; then
    ranks=0
fi

if [[ "$method" = "sgd" ]]; then
    n_trials=25
else
    n_trials=50
fi

gpus=(0 1 2 3 4 5 6)

# --- Configuration ---
for i in "${!ranks[@]}"; do
    rank=${ranks[$i]}
    gpu=${gpus[$i]}
    CUDA_VISIBLE_DEVICES=$gpu \
    python optuna_cifar_trainer.py \
        --wandb_project "cifar10-hpo-2" \
        --batch_size 128 \
        --cifar 10 \
        --seed 42 \
        --resnet_depth 20 \
        --method "$method" \
        --num_workers 7 \
        --use_tnn "$use_tnn" \
        --weight_model_class "$weight_model_class" \
        --rank $rank &
    echo "Started training with rank $rank on GPU $gpu"
done

