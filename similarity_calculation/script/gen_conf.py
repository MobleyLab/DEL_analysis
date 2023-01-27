from openeye import oechem, oeomega
import numpy as np
import pandas as pd
import tools
import argparse
import pickle

def gen_conf(infile):
    '''
    Inputs
    ------
    infile : str
        filename of a .csv file containing SMILES of all compounds to generate conformers for 

    Outputs
    -------
    filename.log : plain text file
       file that records any warning messages from the conformer generation process
       
    filename.pkl : dictionary
        dictionary containing the indices of compounds and their specified enumerated stereoisomers
        
    testname.oeb : file
        OpenEye binary file containing the generated conformers for each compound
    '''
    # parse input file name if only filename is passed in
    filename = infile[:-4]
    # read in input .csv file containing all SMILES strings
    data = pd.read_csv(infile)

    ## Set up error catching
    ## This file will save the error message returned for all compound that do not have specified stereochemistry
    warnfile = filename + '.log'
    fname = warnfile
    errfs = oechem.oeofstream(fname)
    oechem.OEThrow.SetOutputStream(errfs)

    # initialize structures to store information
    new_table = pd.DataFrame(columns=['index', 'Molecule'])
    indices = []
    molecules = []

    # generate conformers for all SMILES strings
    for index, smi in enumerate(data['SMILES']):
        mol = tools.smiles_to_oemol(smi)
        oechem.OETriposAtomNames(mol)
        mol = tools.normalize_molecule(mol)
        cmol = None
        # try to generate conformers for the given molecule
        try:
            cmol = tools.generate_conformers(mol, max_confs=200)
            indices.append(index)
            molecules.append(cmol)

        # will fail if molecule does not have specified stereochemistry
        except Exception as e:
            # write warning message into `warnfile`, which is later parsed to extract molecule indices
            oechem.OEThrow.Warning('Molecule {} returned an error\n{}'.format(index, str(e)))

        # enumerates all possible stereoisomers for a compound with unspecified stereochemistry
        if cmol is None:
            for nmol in oeomega.OEFlipper(mol):
                oechem.OETriposAtomNames(nmol)
                # generate conformers for all valid stereoisomers
                try:
                    nmol = tools.generate_conformers(nmol, max_confs=200)
                    indices.append(index)
                    molecules.append(nmol)
                except Exception:
                    pass

    new_table['index'] = indices
    new_table['Molecule'] = molecules

    # save entire dataframe of generated conformers into database
    outfile = filename + '.oeb'
    tools.write_dataframe_to_file(new_table, outfile)

    ## Record the compounds with enumerated stereoisomers and their indices for later analysis
    ## Create a dictionary where the index of the original compound is the key
    ## and the index of all valid stereoisomers are the values
    groupings = {}
    group = []
    for index, value in enumerate(indices[:-1], 1):
        # start sorting indices if the value of entry is same as the previous
        if indices[index] == indices[index-1]:
            group.append(index-1)
            # edge case to close off group if it occurs at the very end of the list
            if index == len(indices)-1:
                group.append(index)
                groupings[value] = group
        # stop storing indices once value has changed, close off list, append it and make a new list
        elif indices[index] != indices[index-1] and group:
            group.append(index-1)
            groupings[value] = group
            group = []
        else:
            continue

    ## Save dictionary of {original_compound_id: [stereoisomer_compound_ids]} to a file
    ## that will be later used to map stereoisomers back to the original compound
    flagfile = filename + '.pkl'
    pickle.dump(groupings, open(flagfile, 'wb'))

    ## Read error -- will output in std.out if there are issues with any conformers
    ## Otherwise, should output that stereochemistry was enumerated for all compounds
    tools.read_error(warnfile, flagfile)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(
            description="Generate conformers given an input of SMILES strings",
            allow_abbrev=False)

    my_parser.add_argument('--infile',
            action='store',
            type=str,
            help='input .csv file of SMILES',
            required=True)

    args = my_parser.parse_args()
    gen_conf(args.infile)
