#!/usr/bin/env bash
source activate sam
cd /home/cao/code/tensor-bar/td_study

seeds=(42 87 100)
for seed in "${seeds[@]}"; do
    # python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-BAR --alpha 0.0001
    # python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-BAR --alpha 0.0005
    # python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-BAR --alpha 0.001
    python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-BAR --alpha 0.002
    # python run.py --data covid19 --seed $seed --num_iter 1000000 --optimization SGD
    python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-SAM --rho 0.01
    python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-SAM --rho 0.1
    python run.py --data covid19 --seed $seed --num_iter 50000 --optimization ADAM-SAM --rho 0.5
    # python run.py --data covid19 --seed $seed --num_iter 1000000 --optimization Bar --alpha 0.1
    # python.run.py --data covid19 --seed $seed --num_iter 1000000 --optimization Bar --alpha 0.5
    # python.run.py --data covid19 --seed $seed --num_iter 1000000 --optimization Bar --alpha 1
    python run.py --data covid19 --seed $seed --num_iter 50000 --optimization Adam
done