# DEL_analysis
This repository provides relevant code and files to reproduce the analysis from our paper: https://chemrxiv.org/engage/chemrxiv/article-details/6438943f08c86922ffeffe57 


[![DOI](https://zenodo.org/badge/594168758.svg)](https://zenodo.org/badge/latestdoi/594168758)

### Organization
#### Overall layout
Associated code and files for different procedures are separated into distinct directories, outlined below. Each level of organization also contains a separate `README.md` file that better details what can be found. The recommended order for running the code is presented in the Manifest. 

#### Manifest:
- `data_preparation`: clean DEL data and prepare for single building block level analysis
- `similarity_calculation`: calculate 2D/3D Tanimoto similarity of building blocks
- `data_analysis`: analyze the data from an experimental DEL screen
- `prediction`: predict the activity of new compounds using previous experimental information
- `SI`: additional figures and analysis 
- `environment.yml`: environment file
- `figures`: directory of PDFs of all figures in the main text
- `SI_figures`: directory of PDFs of all figures in the Supporting Information

### Requirements
The content here relies primarily on open-source tools and we provide files to reproduce the environment we used to run the analyses. Some materials require an OpenEye license, which is free for academics. 

