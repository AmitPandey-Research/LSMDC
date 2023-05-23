#!/bin/bash
#SBATCH -A amit.pandey
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END

module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2


mkdir -p /scratch/amit/

eval "$(conda shell.bash hook)"
conda activate python38


BASE_DIR="/scratch/amit"

scp ada:/share3/dnaveenr/LSMDC16_info_fillin_new_augmented.json /scratch/amit/

python get_tokenized_captions.py

scp -r /scratch/amit/tokenizer/ ada:/share3/dnaveenr/
scp /scratch/dnaveenr/LSMDC16_info_fillin_sos_eos_blank_mask.json ada:/share3/dnaveenr/
