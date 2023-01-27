#! /usr/bin/bash
#SBATCH --job-name=calculate_tanimoto_combo
#SBATCH --partition=titanx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --gres=gpu:titan:1

#Function to call to run the actual code
source activate oepython

script_path=anagenex/FINAL
cd $script_path

# Initialize variables
dirname='../files'
FILES="${dirname}/*.oeb"

# Create new directory to store all files
for f in "${FILES}"
do
python calc_3D_sim.py --ref "${f}" --test "${f}"
done
