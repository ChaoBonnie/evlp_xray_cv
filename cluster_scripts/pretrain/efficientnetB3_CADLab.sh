#!/bin/bash
#SBATCH --account=rrg-wanglab
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=02-12:00
#SBATCH --output=outputs/%N-%j.out

cd ~/projects/rrg-wanglab/bonchao
source imaging_env/bin/activate
cd evlp_xray_cv
tensorboard --logdir=saved_models/pretrain_nihcxr --host 0.0.0.0 --load_fast false &
python pretrain_nihcxr.py --data_dir ~/scratch/NIH_1024p_png/CADLab --name efficientnetB3_CADLab --cadlab_dataset --model_backbone efficientnetB3