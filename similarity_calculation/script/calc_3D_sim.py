from openeye import oechem, oefastrocs
import numpy as np
import pickle
import os
import sys
import argparse
import clean_3D_sim_matrix
import tools

def calc_3D_sim(ref, test):
    '''
    Inputs
    ------
    ref : str
        filename of the .oeb file containing enumerated stereoisomers and generated conformers for the reference set
    test : str
        filename of the .oeb file containing enumerated stereoisomers and generated conformers for the reference set

    Outputs
    -------
    refname.npy : array
        NumPy array of shape (ref, test) where each element is the highest 3D Tanimoto combo score between a compound
        in the ref set and compound in the test set

    refname.csv : file
        file containing the SMILES of the representative stereoisomer for each compound in the reference set;
        in the case where stereochemistry is specified for that compound, the SMILES and the enumerated_SMILES will be the same

    testname.csv : file
        file containing the SMILES of the representative stereoisomer for each compound in the test set;
        in the case where stereochemistry is specified for that compound, the SMILES and the enumerated_SMILES will be the same
    '''

    refname = ref[:-4]
    testname = test[:-4]

    if not oefastrocs.OEFastROCSIsGPUReady():
        oechem.OEThrow.Info("No supported GPU available!")

    opts = oefastrocs.OEShapeDatabaseOptions()
    opts.SetScoreType(oefastrocs.OEShapeDatabaseType_Combo)

    ifs_0 = oechem.oemolistream()
    ifs_0.open(ref)

    ## Get size of reference set
    ref_size = 0
    for mol in ifs_0.GetOEMols():
        ref_size += 1

    ## Open reference file a second time
    ifs = oechem.oemolistream()
    ifs.open(ref)

    dbase = oefastrocs.OEShapeDatabase()
    moldb = oechem.OEMolDatabase()
    moldb.Open(ifs)

    dots = oechem.OEThreadedDots(10000, 200, 'conformers')
    dbase.Open(moldb, dots)

    qfs_0 = oechem.oemolistream()
    qfs_0.open(test)

    ## Get size of test set
    test_size = 0
    for mol in qfs_0.GetOEMols():
        test_size += 1

    ## Open test file a second time
    qfs = oechem.oemolistream()
    qfs.open(test)

    mcmol = oechem.OEMol()
    qmolidx = 0

    ## We use list of lists instead of numpy array because we want to store
    ## conformer information as a string
    output_matrix = np.zeros((ref_size, test_size))

    ## Loop through all the compounds in the query set
    while oechem.OEReadMolecule(qfs, mcmol):
        ## Looping over each conf of the query molecule
        for q_index, q_conf in enumerate(mcmol.GetConfs()):
            #print('Reading conf {} of {}'.format(index+1, mcmol.NumConfs()))
            for score in dbase.GetSortedScores(q_conf, opts):
                dbmol_idx = score.GetMolIdx()
                #can change to just color or shape Tanimoto
                new_score = score.GetTanimotoCombo()
                if new_score > output_matrix[dbmol_idx][qmolidx]:
                    output_matrix[dbmol_idx][qmolidx] = new_score
        qmolidx += 1

    ## Save matrix prior to cleaning in case there is an issue
    np.save(refname+'.npy', output_matrix)
    print('saved raw matrix')

    ## Clean output matrix
    output_matrix, keep_ind_ref, keep_ind_test = clean_3D_sim_matrix.modify_3D_sim_mat(output_matrix, refname+'.pkl', testname+'.pkl')

    ## Save matrix with modifications
    np.save(refname+'.npy', output_matrix)
    print('overwrote with cleaned matrix')

    ## Save list of SMILES for what each compound is in each index of the matrix
    ref_db = tools.read_file_to_dataframe(ref)
    ref_db.iloc[keep_ind_ref].to_csv(refname+'_row.csv', index=False)

    test_db = tools.read_file_to_dataframe(ref)
    test_db.iloc[keep_ind_test].to_csv(testname+'_col.csv', index=False)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='''
    Calculate 3D similarity matrix of reference and test set
    ''', allow_abbrev=False)

    my_parser.add_argument('--ref',
            action='store',
            type=str,
            help='input .oeb database of reference SMILES',
            required=True)

    my_parser.add_argument('--test',
            action='store',
            type=str,
            help='input .oeb database of test SMILES',
            required=True)

    args = my_parser.parse_args()
    calc_3D_sim(args.ref, args.test)
