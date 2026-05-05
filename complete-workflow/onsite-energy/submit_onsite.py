#!/bin/bash
#SBATCH --job-name=onsite_energy
#SBATCH --array=0-10                     # 31 frames, one task each
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/onsite_%A_%a.out
#SBATCH --error=logs/onsite_%A_%a.err
#SBATCH --time=08:00:00                  # 200 LAMMPS runs per frame
#SBATCH --account=torch_pr_109_general

mkdir -p logs

FRAME=${SLURM_ARRAY_TASK_ID}

echo "Frame ${FRAME} starting on $(hostname) at $(date)"

module load anaconda3/2025.06
# no lammps module needed — using full path in on_site_energy.py

python on_site_energy.py ${FRAME}

echo "Frame ${FRAME} finished at $(date)"
