#!/bin/bash -l
#########################################################
#SBATCH -J BuildDualAGN
#SBATCH --mail-user=spencerlockwood@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /home/stlock/projects/rrg-babul-ad/stlock/slurm_outputs/build_catalog-%j.out
#########################################################
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --array=49-119
#########################################################

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
source /home/stlock/.venvs/romulus_env/bin/activate
# Define output directory
OUTPUT='/scratch/stlock/dualAGNs/outputs/'

# Run the Python script and redirect output
python -u /home/stlock/tripleAGN/scripts/build_dualagn_catalog.py ${SLURM_ARRAY_TASK_ID} &> ${OUTPUT}/output_${SLURM_JOB_ID}.log