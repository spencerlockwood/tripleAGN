#!/bin/bash -l
#########################################################
#SBATCH -J BuildDualAGN
#SBATCH --mail-user=vida.saeidzadeh@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /home/vidasz/scratch/DualAGNs/outputs/build_dualagn-%A_%a.out
#########################################################
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --array=110-120
#########################################################
#49-120
# Load conda environment
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
source /home/vidasz/anaconda3/etc/profile.d/conda.sh
conda activate env_full
# Define output directory
OUTPUT='/home/vidasz/scratch/DualAGNs/outputs/'

# Run the Python script and redirect output
/home/vidasz/anaconda3/envs/env_full/bin/python -u /home/vidasz/scratch/DualAGNs/build_dualagn_catalog.py ${SLURM_ARRAY_TASK_ID} &> ${OUTPUT}/output_${SLURM_JOB_ID}.log
