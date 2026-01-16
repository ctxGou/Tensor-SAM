#!/usr/bin/env bash
source activate sam
cd /home/cao/code/tensor-bar

# --- Configuration ---
GPUs=(0 1 2 3 4 5)
MAX_WORKERS=14

# This variable is now correctly named and used ONLY for the dataloader.
DATALOADER_WORKERS=$((MAX_WORKERS / ${#GPUs[@]})) || true
echo "GPUs available: ${GPUs[@]}"
echo "Dataloader workers per job: $DATALOADER_WORKERS"

# --- Experiment Definitions ---
# Using an array for easier indexing
tuples=(
    # "20, sgd, 0, 1, TensorTrain, 10"
    # "20, sgd, 0, 1, TensorTrain, 20"
    "20, sgd, 0, 1, TensorRing, 10"
    "20, sgd, 0, 1, TensorRing, 20"
    "20, sgd, 0, 1, CP, 50"
    "20, sgd, 0, 1, CP, 100"
    # "20, sgd, 0, 1, Tucker, 10"
    # "20, sgd, 0, 1, Tucker, 20"
    # "20, sgd, 0, 0, None, 0"
)

# --- Function to Launch a Single Training Job ---
# It now logs output to a file instead of /dev/null for easier debugging
launch_job(){
  local gpu=$1 resnet_depth=$2 method=$3 sam_rho=$4 use_tnn=$5 td=$6 rank=$7 workers=$8
  
  # Create a descriptive log file name
  local log_file="logs/gpu${gpu}_${td}_rank${rank}.log"
  echo "  - Log file: $log_file"

  (
    # Each seed runs sequentially within this single background job
    for seed in 41; do
      echo "Starting seed $seed..."
      CUDA_VISIBLE_DEVICES=$gpu \
        python cifar_trainer.py \
            --num_epochs 180 \
            --batch_size 128 \
            --cifar 10 \
            --seed $seed \
            --resnet_depth $resnet_depth \
            --method $method \
            --num_workers "$workers" \
            --sam_rho $sam_rho \
            --use_tnn $use_tnn \
            --weight_model_class "$td" \
            --rank $rank &
    done
  ) > "$log_file" 2>&1 &
}

# --- Main Scheduler Logic ---
mkdir -p logs # Create logs directory
job_queue=( "${tuples[@]}" )
num_jobs=${#job_queue[@]}
jobs_launched=0

# Associative arrays to track which PID is on which GPU
declare -A gpu_pid_map
declare -A pid_gpu_map

# Initialize all GPUs as free
for gpu_id in "${GPUs[@]}"; do
    gpu_pid_map[$gpu_id]=0 # 0 means free
done

echo "---"
echo "Starting job scheduler for $num_jobs experiments..."
echo "---"

while [[ $jobs_launched -lt $num_jobs ]]; do
  # Find a free GPU
  free_gpu=-1
  for gpu_id in "${GPUs[@]}"; do
    if [[ ${gpu_pid_map[$gpu_id]} -eq 0 ]]; then
      free_gpu=$gpu_id
      break
    fi
  done

  # If a GPU is free, launch the next job from the queue
  if [[ $free_gpu -ne -1 ]]; then
    tuple=${job_queue[$jobs_launched]}
    IFS=', ' read -r resnet_depth method sam_rho use_tnn td rank <<< "$tuple"
    
    echo "[$(date +%T)] GPU $free_gpu is free. Launching job $((jobs_launched + 1)) of $num_jobs: ($td, rank $rank)"
    
    launch_job $free_gpu $resnet_depth $method $sam_rho $use_tnn "$td" $rank "$DATALOADER_WORKERS"
    new_pid=$!
    
    # Track the new job
    gpu_pid_map[$free_gpu]=$new_pid
    pid_gpu_map[$new_pid]=$free_gpu
    
    jobs_launched=$((jobs_launched + 1))
    
  else
    # If all GPUs are busy, wait for any one of the running jobs to finish
    echo "[$(date +%T)] All GPUs are busy. Waiting for a job to finish..."
    
    # The 'wait -n' command is perfect for this. It waits for the next job to terminate.
    wait -n -p finished_pid # '-p' stores the finished PID in the variable
    
    # Find out which GPU the finished job was on and mark it as free
    freed_gpu=${pid_gpu_map[$finished_pid]}
    echo "[$(date +%T)] Job with PID $finished_pid on GPU $freed_gpu has finished."
    gpu_pid_map[$freed_gpu]=0
    unset pid_gpu_map[$finished_pid]
  fi
done

echo "---"
echo "All $num_jobs jobs have been launched. Waiting for the last running jobs to complete..."
wait # Wait for all remaining background jobs
echo "All jobs finished successfully."