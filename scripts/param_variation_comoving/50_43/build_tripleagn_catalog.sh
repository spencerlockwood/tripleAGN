#!/bin/bash -l
#########################################################
#SBATCH -J 50_43_BuildTripleAGN
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
#49-120
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
source /home/stlock/.venvs/romulus_env/bin/activate
# Define output directory
OUTPUT='/scratch/stlock/tripleAGNs/outputs/50_43/'

# Run the Python script and redirect output
python -u /home/stlock/tripleAGN/scripts/param_variation_comoving/50_43/build_tripleagn_catalog.py ${SLURM_ARRAY_TASK_ID} &> ${OUTPUT}/output_${SLURM_JOB_ID}.log