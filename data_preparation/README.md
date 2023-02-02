# data_preparation

## What's here:
### input
    - `DEL_hits.csv`: file with SMILES of experimentally determined hits and associated read counts
    - `DEL_inactives.csv`: file with SMILES of experimentally determined inactives

### script
    - `clean.py`: script to clean data and output datasets necessary for further analysis

### output
    - `total_compounds.csv`: file of all cleaned DEL compounds
    - `bb_{1..3}.csv`: files containing SMILES of all unique building blocks for each library position 
