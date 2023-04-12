#! /usr/bin/bash
#SBATCH --job-name=calc_auc_90
##SBATCH --error=python_process.e
#SBATCH --partition=titanx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --gres=gpu:titan:1
#SBATCH --array=0-49
#Function to call to run the actual code
source activate prod_env

SEED=$SLURM_ARRAY_TASK_ID
python scripts/prediction_script.py --seed ${SEED} --frac 0.90
