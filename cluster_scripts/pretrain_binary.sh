#!/bin/bash
#SBATCH --account=rrg-wanglab
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=03-03:00
#SBATCH --output=%N-%j.out

source imaging_env/bin/activate
pip install torch --no-index
pip install pytorch_lightning --no-index
pip install torchvision --no-index
python pretrain_nihcxr.py --data_dir ~/scratch/NIH_images_512p --name first_attempt --binary --debug