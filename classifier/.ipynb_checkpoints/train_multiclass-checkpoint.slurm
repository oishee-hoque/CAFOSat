#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=24:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --gres=gpu:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH -J "binary_classifier"   # job name
#SBATCH --account=nssac_students   # allocation name
#SBATCH --output=slurm_ouputs/set1_resnet18.out
#SBATCH -e slurm_ouputs/set1_resnet18.err

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniforge/24.3.0-py3.11

python /project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/training/multi_train.py