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


eval "$(conda shell.bash hook)"
conda activate tesi_env

echo "======================="
python3 -c "import torch; print('>>> GPU VISTA DA PYTORCH;', torch.cuda.is_available()); print('>>> MODELLO GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nessuna')"
echo "======================"

export MIO_SCRATCH="/scratch_share/bislab/HF_USER_CACHE/$USER"

mkdir -p "$MIO_SCRATCH/modelli_tesi"
mkdir -p job_logs

export HF_HUB_CACHE="$MIO_SCRATCH/modelli_tesi"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_FILE_LOCKS=1
export HF_TOKEN="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"

pwd; hostname; date

python3 -u master_run.py

date
