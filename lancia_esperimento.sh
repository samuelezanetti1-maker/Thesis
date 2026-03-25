#!/bin/bash
#SBATCH --job-name=tesi_zanetti
#SBATCH --partition=only-one-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=job_logs/output_%j.log

module purge
module load amd/gcc-8.5.0/miniforge3

source activate tesi_env

export HF_HOME="/scratch_share/bislab/HF_HUB_CACHE/"
export HUGGINGFACE_HUB_CACHE="/scratch_share/bislab/HF_HUB_CACHE/"
export HF_DATASETS_CACHE="/scratch_share/bislab/HF_XET_CACHE/"

python master_run.py
