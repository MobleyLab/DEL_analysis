# data_preparation

## What's here:
### input
- `del_hits.csv`: file with SMILES of experimentally determined hits and associated read counts
- `del_inactives.csv`: file with SMILES of experimentally determined inactives
- `PG_SMILES.pkl`: contains the SMILES for the protecting groups present in the data
- `deprot_SMIRKS.pkl`: contains SMIRKS patterns to deprotect building blocks 

### script
- `clean.py`: script to clean data and output datasets necessary for further analysis

### output
- `total_compounds.csv`: file of all cleaned DEL compounds
- `bb{1..3}_list.csv`: files containing SMILES of all unique building blocks for each library position

## Procedure
The script `clean.py` runs through all steps we performed to curate our DEL data after experiment.
```python
# curate input files after experiment
python clean.py
``` 
The output includes files of all the unique building blocks for each position of the library, which is needed for similarity calculation.


