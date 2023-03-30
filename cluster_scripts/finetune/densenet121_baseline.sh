#!/bin/bash
#SBATCH --account=rrg-wanglab
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=02-12:00
#SBATCH --output=densenet121_baseline.out

cd ~/projects/rrg-wanglab/bonchao
source imaging_env/bin/activate
cd evlp_xray_cv
tensorboard --logdir=saved_models/pretrain_nihcxr --host 0.0.0.0 --load_fast false &
python finetune_evlp.py --data_dir /cluster/projects/bwanggroup/bchao/datasets/EVLP_CXR/temporal/recipient_outcome/Double/Main --name densenet121_baseline --model_backbone densenet121