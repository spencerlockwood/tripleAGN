#!/bin/bash -l
#########################################################
#SBATCH -J AllBHs
#SBATCH --mail-user=vida.saeidzadeh@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /home/vidasz/projects/rrg-babul-ad/vidasz/mpi_xigrm_z0678-%j.out
#########################################################
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --array=49-119
##SBATCH --constraint=skylake
#########################################################
#[62,65,69,75,80,88,89,96,107,110,120]

# Load conda and activate your environment
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 mpi4py/3.1.4
source /home/vidasz/anaconda3/etc/profile.d/conda.sh
conda activate env_full


DATA_FOLDER='/home/vidasz/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/cosmo25p.768sg1bwK1BHe75.'
OUTPUT='/home/vidasz/scratch/DualAGNs/outputs/'


srun python -u -m mpi4py /home/vidasz/scratch/DualAGNs/Halomap.py ${DATA_FOLDER} ${SLURM_ARRAY_TASK_ID} &> ${OUTPUT}/output_${SLURM_JOB_ID}.log
