#!/bin/bash
#SBATCH -A paditya.sreekar
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/10.0

set -x
python3 my_trainer.py --nc 1 --ny 20 --nz 20 --nt_cond 5 --nt_inf 5 --beta_z 2 --dataset smmnist --data_dir ~/.data --seq_len 15 
