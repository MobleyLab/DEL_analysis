import copy
import argparse
import numpy as np
import pickle

def best_stereoisomer(sim_mat, stereo_groups, ref):
    '''
    Inputs
    ------
    sim_mat : NumPy array
        matrix of 3D similarity scores to be modified
    stereo_groups : dictionary
        dictionary of compound index as key and indices of all its enumerated stereoisomers as value
    ref : bool
        indicates whether input is reference or test set (rows or columns, respectively)

    Returns
    -----
    keep_dict : dictionary
        dictionary of best scoring stereoisomer for each compound
    '''
    keep_dict = {}
    for key in stereo_groups:
        current_score = 0
        for value in stereo_groups[key]:
            if ref:
                score = np.mean(sim_mat[value, :])
            else:
                score = np.mean(sim_mat[:, value])
            # saves only the stereoisomer with highest average similarity score across all compounds
            if score > current_score:
                keep_dict[key] = value
                current_score = score

    return keep_dict

def keep_indices(stereo_groups, keep_groups, N, ref):
    '''
    Inputs
    ------
    stereo_groups : dictionary
        dictionary of compound index as key and indices of all its enumerated stereoisomers as value
    keep_groups : dictionary
        output of `best_stereoisomer`; dictionary of best scoring stereoisomer for each compound
    N : NumPy array
        shape of matrix of 3D similarity scores to be modified
    ref : bool
        indicates whether input is reference or test set (change rows or columns)

    Returns
    -------
    keep_ind : list
        list of all indices to keep
    '''
    total = []
    # Create a list containing only the indices of stereoisomers we wish to remove
    for key in stereo_groups:
        stereo_groups[key].remove(keep_groups[key])
        total.append(stereo_groups[key])
    total = [item for sublist in total for item in sublist]
    # if ref is True, we remove compounds from the rows
    # otherwise we remove compounds from the columns
    if ref:
        keep_ind = list(set(np.arange(N[0])) - set(total))
    else:
        keep_ind = list(set(np.arange(N[1])) - set(total))

    return keep_ind

def modify_3D_sim_mat(sim_mat, ref_groups, test_groups):
    '''
    Inputs
    ------
    sim_mat : NumPy array
        matrix of 3D similarity scores to be modified
    ref_groups : dictionary
        dictionary of reference compound index as key and indices of all its enumerated stereoisomers as value
    test_groups : dictionary
        dictionary of test compound index as key and indices of all its enumerated stereoisomers as value

    Returns
    -------
    new_sim_mat_3D : NumPy array
        matrix of 3D similarity scores with only best scoring stereoisomer for each compound
    '''
    # Load in files
    if type(sim_mat) == str:
        sim_mat = np.load(sim_mat)
    ref_groups = pickle.load(open(ref_groups, 'rb'))
    test_groups = pickle.load(open(test_groups, 'rb'))

    # Make deep copy of dictionaries so the original is not modified
    ref_ind = copy.deepcopy(ref_groups)
    test_ind = copy.deepcopy(test_groups)

    keep_groups_ref = best_stereoisomer(sim_mat, ref_ind, ref=True)
    keep_groups_test = best_stereoisomer(sim_mat, test_ind, ref=False)

    keep_ind_ref = keep_indices(ref_ind, keep_groups_ref, np.shape(sim_mat), ref=True)
    keep_ind_test = keep_indices(test_ind, keep_groups_test, np.shape(sim_mat), ref=False)

    # Slice the rows and columns to keep only one stereoisomer per compound
    new_sim_mat_3D = sim_mat[keep_ind_ref, :][:, keep_ind_test]

    return new_sim_mat_3D, keep_ind_ref, keep_ind_test

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(
    description="Remove additional compound stereoisomers from 3D similarity score matrix",
    allow_abbrev=False)

    my_parser.add_argument('--matrix',
            action='store',
            type=str,
            help='path to 3D similarity matrix',
            required=True)
    my_parser.add_argument('--ref_group',
            action='store',
            type=str,
            help='path to dictionary of reference groups',
            required=True)
    my_parser.add_argument('--test_group',
            action='store',
            type=str,
            help='path to dictionary of test groups',
            required=True)

    args = my_parser.parse_args()

    new_mat = modify_3D_sim_mat(args.matrix, args.ref_group, args.test_group)
    np.save('{}_mod.npy'.format(args.matrix[:-4]), new_mat)
