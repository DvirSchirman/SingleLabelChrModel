#!/bin/bash

#SBATCH -J droppedRandom_9
#SBATCH -o sim_out/%J-%a.o
#SBATCH -e sim_out/%J-%a.o
#SBATCH -A p2018001
#SBATCH -t 4:00:00
#SBATCH -n 20
#SBATCH -M rackham

module load conda
source conda_init.sh
conda activate openmm-env
PATH=~/.conda/envs/openmm-env/bin/:$PATH
export PATH
cd /crex/proj/uppstore2018129/elflab/PersonalFolders/Dvir/Chromosome_structure/Modeling/polychrom/localization_based_sim
#modelsPath="241022_4Paper_normalModel"
#modelsPath="241022_4Paper_noRestraints"
#modelsPath="241022_4Paper_randomAll_20241022" 

#modelsPath="241022_4Paper_droppedInput_9" 
modelsPath="241022_4Paper_droppedInput_randomAll_20241022_9"

#python getForkPlots.py -m $modelsPath
python getForkPlotsDroppedLoci.py -m $modelsPath
#python getBulkDistanceMat.py -m $modelsPath
#python getBulkDistanceMat_CisTrans.py -m $modelsPath
#python getDynamicDistanceMat.py -m $modelsPath
#python getDynamicDistanceMat_CisTrans.py -m $modelsPath
#python localization_based_model_constrained.py -n 1 -o 240527_test_constrained1 --use_cpu

#python getBulkDistanceMatNInstancesBatch.py -m $1 -n 10
