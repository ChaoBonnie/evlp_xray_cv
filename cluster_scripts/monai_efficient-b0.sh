#!/bin/bash
#SBATCH --account=rrg-wanglab
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=03-03:00
#SBATCH --output=%N-%j.out

cd ~
source monai_env/bin/activate
cd ~/projects/rrg-wanglab/bonchao/evlp_xray_cv
tensorboard --logdir=saved_models --host 0.0.0.0 --load_fast false &
python gpu_train_classify.py