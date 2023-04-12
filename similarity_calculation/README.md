# similarity_calculation

## What's here:
### script
- `gen_conf.py`: script to generate conformers for a set of input SMILES using OpenEye's OMEGA
- `calc_2D_sim.py`: script to calculate 2D Tanimoto scores using RDKit
- `calc_3D_sim.py`: script to calculate 3D Tanimoto combo using OpenEye's FastROCS
- `clean_3D_sim_matrix.py`: script to select the best scoring stereoisomer for molecules with unspecified stereochemistry

### output
- `bb{1..3}_list.oeb`: OpenEye binary file of generated conformers for the building blocks at each position
- `bb{1..3}_list.npy`: arrays containing pairwise 3D Tanimoto combo scores for each building block position

## Procedure
Following data curation, we generate conformers for the building blocks at each position of the library. These conformers are then used to calculate 3D Tanimoto combo scores for each building block position. This procedure needs to be repeated for each position of the library.
  
```python
# Generate conformers
python gen_conf.py --infile "bb_list.csv"

# Calculate 3D similarity
python calc_3D_sim.py --ref "bb_list.oeb" --test "bb_list.oeb"
```
The output includes OpenEye binary files which store all the generated conformers for each building block and NumPy arrays with the 3D Tanimoto combo scores for the building blocks in each position, the latter of which is needed for data analysis.

We also provide code to calculate 2D Tanimoto similarity should 3D Tanimoto be inaccessible, but highly recommend using 3D if possible.  
