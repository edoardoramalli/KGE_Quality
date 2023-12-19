#!/bin/bash
#SBATCH -A IscrC_XAI
#SBATCH -p dgx_usr_preempt
#SBATCH --time 07:50:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 tasks
#SBATCH --gres=gpu:1        # 1 gpus per node out of 8
#SBATCH --mem=64000         # memory per node out of 980000 MB
#SBATCH --output=JOB.%j.out_slurm

cd /dgx_scratch/userexternal/eramalli/KGE_Quality/Training && /dgx/home/userexternal/eramalli/miniconda3/envs/kge/bin/python /dgx_scratch/userexternal/eramalli/KGE_Quality/Training/consumeTask.py