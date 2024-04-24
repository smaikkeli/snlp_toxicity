#!/bin/bash -l
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-cpu=1G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu
module load miniconda
source activate snlp
###########################################################
python3 pipeline.py
