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
    ranks=(10 20 50 100 200) # (10 20 50 100 200) # CP
fi
if [ "$weight_model_class" = "TensorRing" ] ; then
    ranks=(5 7 12 17 23) # TensorRing
    # ranks=(17 23)
fi
if [ "$weight_model_class" = "Tucker" ] ; then
    ranks=(6 9 13 21 24) # Tucker
fi
if [ "$use_tnn" = "0" ] ; then
    ranks=0
fi

cifar=10
resnet_depth=20

# hpo_config= ./results/cifar{cifar}/resnet{resnet_depth}/{weight_model_class}/{rank}/{method}/hpo_{weight_model_class}_rank{rank}_{method}_results.txt


# cifar, resnet_depth, method, sam_rho, gbar_alpha, gbar_alpha_scheduler, weight_model_class, rank, lr, hpo_config
# generate parameters, set sam_rho=-1, gbar_alpha=-1
params=()
sam_rho=-1
gbar_alpha=-1
lr=0.1
gbar_alpha_scheduler="constant"
for rank in "${ranks[@]}"; do
    hpo_config="./results/cifar${cifar}/resnet${resnet_depth}/${weight_model_class}/${rank}/${method}/hpo_${weight_model_class}_rank${rank}_${method}_results.txt"
    # if method is "sgd", set hpo_config to "None"
    if [[ "$method" == "sgd" ]]; then
        hpo_config="None"
    fi
    if [[ "$method" == "balancedinit" ]]; then
        hpo_config="None"
    fi
    params+=("$cifar,$resnet_depth,$method,$sam_rho,$gbar_alpha,$gbar_alpha_scheduler,$weight_model_class,$rank,$lr,$hpo_config")
done


repeats=10

extended_params=()
for param in "${params[@]}"; do
    for ((i=0; i<repeats; i++)); do
        extended_params+=("$param")
    done
done

gpus=(0 1 2 3 4)

launch_job() {
    local gpu_id=$1
    local seed=$2
    local cifar=$3
    local resnet_depth=$4
    local method=$5
    local sam_rho=$6
    local gbar_alpha=$7
    local gbar_alpha_scheduler=$8
    local weight_model_class=$9
    local rank=${10}
    local lr=${11}
    local hpo_config=${12}
    (

    echo "Launching job on GPU $gpu_id with params:"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python cifar_trainer.py \
        --wandb_project "cifar10-CP-best-HP-2" \
        --num_epochs 180 \
        --batch_size 128 \
        --cifar $cifar \
        --seed $seed \
        --resnet_depth $resnet_depth \
        --lr $lr \
        --method $method \
        --num_workers 5 \
        --sam_rho $sam_rho \
        --gbar_alpha $gbar_alpha \
        --gbar_alpha_scheduler $gbar_alpha_scheduler \
        --use_tnn 1 \
        --weight_model_class "$weight_model_class" \
        --rank $rank \
        --hpo_config "$hpo_config" \
    )
}

is_gpu_free() {
    local gpu_id=$1
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu_id + 1))p")
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sed -n "$((gpu_id + 1))p")
    if (( memory_used > 300 || utilization > 10 )); then
        return 1 # GPU is occupied
    else
        return 0 # GPU is free
    fi
}

for gpu_id in "${gpus[@]}"; do
    # Check if the GPU is free at the start
    if is_gpu_free $gpu_id; then
        echo "GPU $gpu_id is free."
    else
        echo "GPU $gpu_id is occupied."
    fi
done

# Main loop to launch jobs
seed=87  # Initial seed for reproducibility
for param in "${extended_params[@]}"; do
    while true; do
        for gpu_id in "${gpus[@]}"; do
            if is_gpu_free $gpu_id; then
                # Parse the parameters
                param=$(echo "$param" | sed 's/, */,/g')
                IFS=',' read -r cifar resnet_depth method sam_rho gbar_alpha gbar_alpha_scheduler weight_model_class rank lr hpo_config <<< "$param"

                # Launch the job on the free GPU
                echo "Launching job on GPU $gpu_id with params: $param"
               launch_job $gpu_id $seed $cifar $resnet_depth $method $sam_rho $gbar_alpha $gbar_alpha_scheduler $weight_model_class $rank $lr "$hpo_config" &

                sleep 30

                # Increment the seed for the next job
                seed=$((seed + 1))

                # Break out of the GPU loop to move to the next job
                break 2
            fi
        done

        # Wait for a short time before checking again if all GPUs are occupied
        sleep 5
    done
done

# Wait for all background jobs to finish
wait