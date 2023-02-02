#! /usr/bin/bash
#SBATCH --job-name=calculate_tanimoto_combo
#SBATCH --partition=titanx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --gres=gpu:titan:1

# Load in appropriate working environment
source activate oepython

# Generate conformers
python gen_conf.py --infile "building_blocks.csv"

# Calculate 3D similarity
python calc_3D_sim.py --ref "building_blocks.oeb" --test "building_blocks.oeb"
