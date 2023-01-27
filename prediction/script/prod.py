import argparse
import hdbscan
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn import tree
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, precision_recall_curve, pairwise_distances
import umap
import warnings

def customwarn(message, category, filename, lineno, file=None, line=None):
    warning_file = open("warn.txt", "a")
    warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
    warning_file.close()

def unique_bb(df, bb_pos):
    return df[bb_pos].unique()

def sample_bb(unique_bbs, bb_pos, frac, seed):
    np.random.seed(seed)
    train_ind = np.random.choice(np.arange(len(unique_bbs)), size=int(frac*len(unique_bbs)), replace=False)
    test_ind = np.setdiff1d(np.arange(len(unique_bbs)), train_ind)
    train_bbs = unique_bbs[train_ind]
    test_bbs = unique_bbs[test_ind]
    return pd.DataFrame(train_bbs, columns=[bb_pos]), pd.DataFrame(test_bbs, columns=[bb_pos])

def train_test_set(df, bb1_train, bb2_train, bb3_train):
    train = df.merge(bb1_train, on='bb1').merge(bb2_train, on='bb2').merge(bb3_train, on='bb3')
    itd = pd.merge(df, train['structure'], how='left', on='structure', indicator=True)
    test = itd.loc[itd['_merge'] == 'left_only'].drop(columns='_merge')
    return train, test

def update_bbs(train, bb_train, bb_test, bb_pos):
    new_bbs = unique_bb(train, bb_pos)
    old_bbs = bb_train[bb_pos]
    bb_update = list(set(old_bbs) - set(new_bbs))
    return pd.DataFrame(new_bbs, columns=[bb_pos]), pd.concat([bb_test, pd.Series(bb_update, name=bb_pos).to_frame()], ignore_index=True)

def calc_pactive(data, bb_train, bb_pos):
    data['active'] = [0 if x == 0 else 1 for x in data['read_count']]
    val = data.groupby([bb_pos], as_index=False)['active'].mean()
    merged = pd.merge(bb_train, val, on=bb_pos).rename(columns={'active': 'P(active)'})
    return merged

def gen_ecfp6(SMILES):
    '''
    Generates ECFP6 fingerprint for the inputted molecule SMILES

    Input
    -----
    SMILES : str
        SMILES string for the compound of interest

    Output
    ------
    set_fps : RDKit sparse vector
        ECFP fingerprint for the compound
    '''
    if type(SMILES) == str:
        mol = Chem.MolFromSmiles(SMILES)
        set_fps = AllChem.GetMorganFingerprint(mol, 3, useCounts=False)
    else:
        set_mols = [Chem.MolFromSmiles(smi) for smi in SMILES]
        set_fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=False) for mol in set_mols]
    return set_fps

def ecfp6_tanimoto_matrix(row_fps, column_fps):
    '''
    Calculates pairwise matrix of Tanimoto scores for the fingerprints provided

    Input
    -----
    row_fps : list
        list of molecular fingerprints corresponding to the rows of the output matrix
    column_fps : list
        list of molecular fingerprints corresponding to the columns of the output matrix

    Output
    ------
    matrix : array
        matrix containing the Tanimoto score of each pair of compounds
    '''
    matrix = np.zeros((len(row_fps), len(column_fps)))
    for i in range(len(row_fps)):
        for j in range(len(column_fps)):
            matrix[i][j] = DataStructs.TanimotoSimilarity(row_fps[i], column_fps[j])
    return matrix

def calc_dist_mat(row_SMILES, col_SMILES):
    bb_row_fps = [gen_ecfp6(smi) for smi in row_SMILES]
    bb_col_fps = [gen_ecfp6(smi) for smi in col_SMILES]
    bb_sim = ecfp6_tanimoto_matrix(bb_row_fps, bb_col_fps)
    return 1 - bb_sim

def UMAP_dist(dist_mat, seed):
    U = umap.UMAP(random_state=seed, metric='precomputed')
    V = U.fit(dist_mat)
    return V.embedding_, V

def assign_coords(bb_df, umap_dist):
    bb_df['X'] = umap_dist[:, 0]
    bb_df['Y'] = umap_dist[:, 1]
    return bb_df

def intracluster_dist(df):
    dist_mat = pairwise_distances(df[['X', 'Y']])
    N = len(dist_mat)
    ind = np.triu_indices(N, k=1)
    return np.mean(dist_mat[ind])

def obj(params):
    return params['n_noise'] + 10*params['icd']

def optimal_hdbscan(bb_df, min_cluster=(3,61), min_samples=(1,21)):
    BB = bb_df.copy(deep=True)
    hdb_params = pd.DataFrame(columns=['min_cluster_size', 'min_samples', 'n_noise', 'icd'])
    for i in min_cluster:
        for j in min_samples:
            if i <= j:
                pass
            else:
                info = {}
                coords = BB[['X','Y']].to_numpy()
                cluster = hdbscan.HDBSCAN(min_cluster_size=int(i), min_samples=int(j), metric='euclidean', gen_min_span_tree=True,
                                  allow_single_cluster=False).fit(coords)

                BB['Cluster'] = cluster.labels_
                info['min_cluster_size'] = int(i)
                info['min_samples'] = int(j)
                info['n_noise'] = len(BB.loc[BB['Cluster'] == -1])

                g = BB.groupby('Cluster')
                h = g.apply(intracluster_dist).reset_index(name='dist')
                info['icd'] = np.mean(h.loc[h['Cluster'] > -1, 'dist'])

                hdb_params = hdb_params.append(info, ignore_index=True)

    ind = np.argmin(obj(hdb_params))
    opt = hdb_params.iloc[ind]
    return (int(opt['min_cluster_size']), int(opt['min_samples']))

def cluster(bb_df, opt_params):
    coords = bb_df[['X','Y']].to_numpy()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=opt_params[0], min_samples=opt_params[1], metric='euclidean',
                              gen_min_span_tree=True, allow_single_cluster=False, prediction_data=True).fit(coords)

    return clusterer, clusterer.labels_

def predict_cluster(bb_umap, bb_clusterer, dist_mat, train_ind, test_ind):
    test_coords = bb_umap.transform(dist_mat[np.ix_(test_ind, train_ind)])
    bb_cluster, bb_prob = hdbscan.prediction.approximate_predict(bb_clusterer, test_coords)
    return test_coords, bb_cluster, bb_prob

def get_cluster_nn(test_bb, bb_pos, all_bb, bb_train, dist_mat):
    bb_SMILES = test_bb[bb_pos]
    bb_id = np.where(all_bb == bb_SMILES)[0]
    cluster_id = test_bb['Cluster']
    cluster_bb_ids = list(bb_train.loc[bb_train['Cluster'] == cluster_id].index)
    ind = np.ix_(bb_id, cluster_bb_ids)
    nn_ind = np.argmin(dist_mat[ind])
    cluster_nn_ind = cluster_bb_ids[nn_ind]
    return bb_train.iloc[cluster_nn_ind]['P(active)']

def get_cluster_med(test_bb, bb_train):
    cluster_id = test_bb['Cluster']
    return np.median(bb_train.loc[bb_train['Cluster'] == cluster_id, 'P(active)'])

def get_cluster_mean(test_bb, bb_train):
    cluster_id = test_bb['Cluster']
    return np.mean(bb_train.loc[bb_train['Cluster'] == cluster_id, 'P(active)'])

def get_cluster_random(bb_train, bb_test, seed):
   # np.random.seed(seed)
    cluster_id, cluster_freq = np.unique(bb_test['Cluster'], return_counts=True)
    pred = np.ones(len(bb_test))*-1
    for val, count in zip(cluster_id, cluster_freq):
        ind = list(bb_test.loc[bb_test['Cluster'] == val].index)
        compounds = bb_train.loc[bb_train['Cluster'] == val]
        assign_id = compounds.sample(n=count, replace=True)['P(active)'].values
        pred[ind] = assign_id
    return pred

def get_test_pred(bb_test, bb_pos, all_bb, bb_train, dist_mat, seed, how='nn'):
    np.random.seed(seed)
    if how == 'nn':
        bb_pred = [get_cluster_nn(bb_test.iloc[x], bb_pos, all_bb, bb_train, dist_mat) for x in np.arange(len(bb_test))]
    elif how == 'med':
        bb_pred = [get_cluster_med(bb_test.iloc[x], bb_train) for x in np.arange(len(bb_test))]
    elif how == 'mean':
        bb_pred = [get_cluster_mean(bb_test.iloc[x], bb_train) for x in np.arange(len(bb_test))]
    elif how == 'crand':
        bb_pred = get_cluster_random(bb_train, bb_test, seed)
    elif how == 'random':
        bb_pred = np.random.choice(bb_train['P(active)'], size=len(bb_test))
    return bb_pred

def format_data_for_tree(data, bb1_data, bb2_data, bb3_data):
    merged = data.merge(bb1_data[['bb1', 'P(active)', 'Cluster']], on='bb1').rename(columns={'P(active)': 'P(active)_1', 'Cluster': 'bb1_Cluster'})\
    .merge(bb2_data[['bb2', 'P(active)', 'Cluster']], on='bb2').rename(columns={'P(active)': 'P(active)_2', 'Cluster': 'bb2_Cluster'})\
    .merge(bb3_data[['bb3', 'P(active)', 'Cluster']], on='bb3').rename(columns={'P(active)': 'P(active)_3', 'Cluster': 'bb3_Cluster'})
    merged['active'] = [0 if x == 0 else 1 for x in merged['read_count']]
    return merged

def create_tree(train, seed, depth=5):
    train_features = train[['bb1_Cluster', 'bb2_Cluster', 'bb3_Cluster', 'P(active)_1', 'P(active)_2', 'P(active)_3']]
    train_targets = train['active']
    decision_tree = tree.DecisionTreeClassifier(random_state=seed, max_depth=depth)
    decision_tree = decision_tree.fit(train_features, train_targets)
    return decision_tree

def predict_activity(data, tree):
    features = data[['bb1_Cluster', 'bb2_Cluster', 'bb3_Cluster', 'P(active)_1', 'P(active)_2', 'P(active)_3']]
    return tree.predict_proba(features)[:,1]

def calculate_auc(targets, probs):
    precision, recall, threshold = precision_recall_curve(targets, probs)
    return auc(recall, precision)

def parse_args():
    my_parser = argparse.ArgumentParser(description="calculates AUC of model")
    my_parser.add_argument('--seed', type=int, help='random seed with which to initialize run', required=True)
    args = my_parser.parse_args()
    return args

def main():
    warnings.showwarning = customwarn
    args = parse_args()
    seed_val = args.seed
    DF = pd.read_csv('../data_preparation/output/total_compounds.csv')
    df = DF.sample(n=100000, random_state=10)

    all_bb1 = unique_bb(df, bb_pos='bb1')
    all_bb2 = unique_bb(df, bb_pos='bb2')
    all_bb3 = unique_bb(df, bb_pos='bb3')

    bb1_train, bb1_test = sample_bb(all_bb1, bb_pos='bb1', frac=0.95, seed=seed_val)
    bb2_train, bb2_test = sample_bb(all_bb2, bb_pos='bb2', frac=0.95, seed=seed_val)
    bb3_train, bb3_test = sample_bb(all_bb3, bb_pos='bb3', frac=0.95, seed=seed_val)

    train, test = train_test_set(df, bb1_train, bb2_train, bb3_train)

    bb1_train, bb1_test = update_bbs(train, bb1_train, bb1_test, 'bb1')
    bb2_train, bb2_test = update_bbs(train, bb2_train, bb2_test, 'bb2')
    bb3_train, bb3_test = update_bbs(train, bb3_train, bb3_test, 'bb3')

    bb1_train = calc_pactive(train, bb1_train, bb_pos='bb1')
    bb2_train = calc_pactive(train, bb2_train, bb_pos='bb2')
    bb3_train = calc_pactive(train, bb3_train, bb_pos='bb3')

    all_bb1 = pd.concat([bb1_train['bb1'], bb1_test['bb1']]).values
    all_bb2 = pd.concat([bb2_train['bb2'], bb2_test['bb2']]).values
    all_bb3 = pd.concat([bb3_train['bb3'], bb3_test['bb3']]).values

    bb1_train_ind = np.arange(len(bb1_train['bb1']))
    bb2_train_ind = np.arange(len(bb2_train['bb2']))
    bb3_train_ind = np.arange(len(bb3_train['bb3']))

    bb1_test_ind = np.arange(len(bb1_train['bb1']), len(all_bb1))
    bb2_test_ind = np.arange(len(bb2_train['bb2']), len(all_bb2))
    bb3_test_ind = np.arange(len(bb3_train['bb3']), len(all_bb3))

    bb1_dist = calc_dist_mat(all_bb1, all_bb1)
    bb2_dist = calc_dist_mat(all_bb2, all_bb2)
    bb3_dist = calc_dist_mat(all_bb3, all_bb3)

    bb1_coords, bb1_umap = UMAP_dist(bb1_dist[np.ix_(bb1_train_ind, bb1_train_ind)], seed=seed_val)
    bb2_coords, bb2_umap = UMAP_dist(bb2_dist[np.ix_(bb2_train_ind, bb2_train_ind)], seed=seed_val)
    bb3_coords, bb3_umap = UMAP_dist(bb3_dist[np.ix_(bb3_train_ind, bb3_train_ind)], seed=seed_val)

    bb1_train = assign_coords(bb1_train, bb1_coords)
    bb2_train = assign_coords(bb2_train, bb2_coords)
    bb3_train = assign_coords(bb3_train, bb3_coords)

    opt_params_bb1 = optimal_hdbscan(bb1_train)
    opt_params_bb2 = optimal_hdbscan(bb2_train)
    opt_params_bb3 = optimal_hdbscan(bb3_train)

    bb1_clusterer, bb1_cluster_labels = cluster(bb1_train, opt_params_bb1)
    bb2_clusterer, bb2_cluster_labels = cluster(bb2_train, opt_params_bb2)
    bb3_clusterer, bb3_cluster_labels = cluster(bb3_train, opt_params_bb3)

    bb1_train['Cluster'] = bb1_cluster_labels
    bb2_train['Cluster'] = bb2_cluster_labels
    bb3_train['Cluster'] = bb3_cluster_labels

    bb1_test_coords, bb1_cluster, bb1_prob = predict_cluster(bb1_umap, bb1_clusterer, bb1_dist, bb1_train_ind, bb1_test_ind)
    bb2_test_coords, bb2_cluster, bb2_prob = predict_cluster(bb2_umap, bb2_clusterer, bb2_dist, bb2_train_ind, bb2_test_ind)
    bb3_test_coords, bb3_cluster, bb3_prob = predict_cluster(bb3_umap, bb3_clusterer, bb3_dist, bb3_train_ind, bb3_test_ind)

    bb1_test = assign_coords(bb1_test, bb1_test_coords)
    bb2_test = assign_coords(bb2_test, bb2_test_coords)
    bb3_test = assign_coords(bb3_test, bb3_test_coords)

    bb1_test['Cluster'] = bb1_cluster
    bb2_test['Cluster'] = bb2_cluster
    bb3_test['Cluster'] = bb3_cluster

    method = ['nn', 'med', 'mean', 'crand', 'random']
    ans = pd.DataFrame(columns=method)
    values = []
    for met in method:
        bb1_test['P(active)'] = get_test_pred(bb1_test, 'bb1', all_bb1, bb1_train, bb1_dist, how=met, seed=seed_val)
        bb2_test['P(active)'] = get_test_pred(bb2_test, 'bb2', all_bb2, bb2_train, bb2_dist, how=met, seed=seed_val)
        bb3_test['P(active)'] = get_test_pred(bb3_test, 'bb3', all_bb3, bb3_train, bb3_dist, how=met, seed=seed_val)

        bb1_comb = pd.concat([bb1_train, bb1_test])
        bb2_comb = pd.concat([bb2_train, bb2_test])
        bb3_comb = pd.concat([bb3_train, bb3_test])

        test_input = format_data_for_tree(test, bb1_comb, bb2_comb, bb3_comb)
        tree_classifier = create_tree(test_input, seed=seed_val, depth=5)
        test_prob = predict_activity(test_input, tree_classifier)
        val = calculate_auc(test_input['active'], test_prob)
        values.append(val)

    ans.loc[len(ans)] = values
    #ans.to_csv('values.csv', mode='a', index=False, header=True)
    print(ans)
    return ans

if __name__ == "__main__":
    main()
