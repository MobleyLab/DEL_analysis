# similarity_calculation

## What's here:
### input
- `bb_{1..3}_list.csv`: SMILES of all unique building blocks for each library position 

### script
- `gen_conf.py`: script to generate conformers for a set of input SMILES using OpenEye's OMEGA
- `calc_3D_sim.py`: script to calculate 3D Tanimoto combo using OpenEye's FastROCS
- `clean_3D_sim_matrix.py`: script to select the best scoring stereoisomer for molecules with unspecified stereochemistry
- `run_3D.sh`: sample job submission script

### output
- `bb_{1..3}_list.npy`: arrays containing pairwise 3D Tanimoto combo scores for each building block position
