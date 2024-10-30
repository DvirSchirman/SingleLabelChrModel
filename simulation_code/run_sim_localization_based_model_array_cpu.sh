#!/bin/bash

#SBATCH -J randomDroppedInput_0
#SBATCH --array=1-150
#SBATCH -o sim_out/%J-%a.o
#SBATCH -A p2018001
#SBATCH -t 72:00:00
#SBATCH -n 1

timestamp=$(date +%s)
module load conda
source conda_init.sh
conda activate openmm-env
PATH=~/.conda/envs/openmm-env/bin/:$PATH
export PATH
echo $seed
python localization_based_model_constrained.py -o 241022_4Paper_droppedInput -d 0 -n 150 -w 150  -i ${SLURM_ARRAY_TASK_ID} --use_cpu --randomize_all -s 20241022

