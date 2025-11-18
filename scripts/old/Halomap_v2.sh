#!/bin/bash -l
#########################################################
#SBATCH -J AllBHs
#SBATCH --mail-user=spencerlockwood@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /home/stlock/projects/rrg-babul-ad/stlock/slurm_outputs/halomap-%j.out
#########################################################
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --array=0-119
##SBATCH --constraint=skylake
#########################################################
##[62,65,69,75,80,88,89,96,107,110,120]

# Load conda and activate your environment
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 mpi4py/3.1.4
source /home/stlock/.venvs/romulus_env/bin/activate


DATA_FOLDER='/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/cosmo25p.768sg1bwK1BHe75.'
OUTPUT='/scratch/stlock/dualagns/outputs/'



srun python -u -m mpi4py /home/stlock/tripleAGN/scripts/Halomap_v2.py ${DATA_FOLDER} ${SLURM_ARRAY_TASK_ID} &> ${OUTPUT}/output_${SLURM_JOB_ID}.log