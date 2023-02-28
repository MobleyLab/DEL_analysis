import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from IPython.display import display, Image
import mols2grid
import umap
import hdbscan
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from scipy.spatial import distance_matrix
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, pairwise_distances, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score

def calc_pactive(data, bb_pos):
    data['active'] = [0 if x == 0 else 1 for x in data['read_count']]
    val = data.groupby([bb_pos], as_index=False)['active'].mean().rename(columns={'active': 'P(active)'})
    return val

def set_bins(bb_pactive, bins):
    bb_bins = bins
    bb_ticks = (bb_bins[:-1] + bb_bins[1:])/2
    bb_labels = [f'[{bb_bins[i]:.2f}, {bb_bins[i+1]:.2f})' for i in range(len(bb_bins)-1)]
    info = np.histogram(bb_pactive, bb_bins)
    return info, bb_ticks, bb_labels

def plot_pactive(bb1_pactive, bb2_pactive, bb3_pactive):
    all_bins = np.arange(0, 1.13, 0.13)
    bb1_bins = np.linspace(np.min(bb1_pactive['P(active)']), np.max(bb1_pactive['P(active)']), 5)
    a, a_ticks, a_labels = set_bins(bb1_pactive['P(active)'], bins=all_bins)
    a_labels[-1] = '[0.91, 1.00)'

    bb2_bins = np.linspace(np.min(bb2_pactive['P(active)']), np.max(bb2_pactive['P(active)']), 5)
    b, b_ticks, b_labels = set_bins(bb2_pactive['P(active)'], bins=all_bins)
    b_labels[-1] = '[0.91, 1.00)'

    bb3_bins = np.linspace(np.min(bb3_pactive['P(active)']), np.max(bb3_pactive['P(active)']), 9)
    c, c_ticks, c_labels = set_bins(bb3_pactive['P(active)'], bins=all_bins)
    c_labels[-1] = '[0.91, 1.00)'
    
    fig, axs = plt.subplots(3, 1, figsize=(10,14), dpi=150, sharey=True)
    plt.subplots_adjust(hspace=0.15)
    a_bars = axs[0].bar(a_ticks, height=a[0], width=0.06, log=True, color='blue')
    axs[0].bar_label(a_bars, fontsize=18, padding=0)
    axs[0].set_xticks([])
    axs[0].set_yticks(np.array([1, 10, 100, 1000, 10000]))
    axs[0].set_yticklabels(np.array([1, 10, 100, 1000, 10000]), fontsize=20)
    axs[0].set_xlim([0, 1])

    b_bars = axs[1].bar(b_ticks, height=b[0], width=0.06, log=True, color='orange')
    axs[1].bar_label(b_bars, fontsize=18, padding=0)
    axs[1].set_xticks([])
    axs[1].set_yticks(np.array([1, 10, 100, 1000, 10000]))
    axs[1].set_yticklabels(np.array([1, 10, 100, 1000, 10000]), fontsize=20)
    axs[1].set_xlim([0, 1])
    axs[1].set_ylabel('number of building blocks', fontsize=28, labelpad=20)

    c_bars = axs[2].bar(c_ticks, height=c[0], width=0.06, log=True, color='green')
    axs[2].bar_label(c_bars, fontsize=18, padding=0)
    axs[2].set_xticks(c_ticks)
    axs[2].set_xticklabels(c_labels, rotation=25, ha='right', fontsize=20)
    axs[2].set_yticks(np.array([1, 10, 100, 1000, 10000]))
    axs[2].set_yticklabels(labels=['$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$10^{4}$'], fontsize=20)
    axs[2].set_xlim([0, 1])
    axs[2].set_ylim([1, 10000])
    axs[2].set_xlabel('building block P(active)', fontsize=28, labelpad=15)
    
    return a[0], b[0], c[0]
    
def view_top_bbs(bb_pactive, bb_pos, N):
    bb_sorted = bb_pactive.sort_values(by='P(active)', ascending=False)
    bb_top = bb_sorted[:N]
    bb_mols = [Chem.MolFromSmiles(smi) for smi in bb_top[bb_pos]]
    img = Draw.MolsToGridImage(bb_mols, molsPerRow=N, returnPNG=False,
                               legends=[f'P(active): {x:.3f}' for x in bb_top['P(active)']])
    return img

def merge_df(df, bb1_pactive, bb2_pactive, bb3_pactive):
    return df.merge(bb1_pactive, on='bb1')\
        .rename(columns={'P(active)': 'P(active)_1'})\
        .merge(bb2_pactive, on='bb2')\
        .rename(columns={'P(active)': 'P(active)_2'})\
        .merge(bb3_pactive, on='bb3')\
        .rename(columns={'P(active)': 'P(active)_3'})

def get_actives(total):
    total['active'] = [0 if x == 0 else 1 for x in total['read_count']]
    actives = total.loc[total['active'] == 1]
    return actives

def plot_compatible(total_actives):
    D_12 = total_actives.groupby(['bb1', 'P(active)_1'], as_index=False)['bb2'].nunique()
    D_13 = total_actives.groupby(['bb1', 'P(active)_1'], as_index=False)['bb3'].nunique()

    D_21 = total_actives.groupby(['bb2', 'P(active)_2'], as_index=False)['bb1'].nunique()
    D_23 = total_actives.groupby(['bb2', 'P(active)_2'], as_index=False)['bb3'].nunique()

    D_31 = total_actives.groupby(['bb3', 'P(active)_3'], as_index=False)['bb1'].nunique()
    D_32 = total_actives.groupby(['bb3', 'P(active)_3'], as_index=False)['bb2'].nunique()

    fig, axs = plt.subplots(3, 3, figsize=(20, 20), dpi=150, sharex=True)
    plt.subplots_adjust(wspace=0.12, hspace=0.05)

    axs[0][1].scatter(D_21['P(active)_2'], D_21['bb1'], color='orange')
    axs[0][2].scatter(D_31['P(active)_3'], D_31['bb1'], color='green')
    axs[0][0].set_xticks([])
    axs[0][1].set_xticks([])
    axs[0][2].set_xticks([])
    p1_ticks = np.arange(0, 600, 100)
    axs[0][0].set_ylim([-0.05, 1.05])
    axs[0][0].set_ylabel('compatible BBs in $p_{1}$', fontsize=24, labelpad=18)
    axs[0][0].set_yticks(np.arange(0, 1.2, 0.2))
    axs[0][0].set_yticklabels(labels=p1_ticks, fontsize=18)
    axs[0][1].set_yticks([])
    axs[0][2].set_yticks([])

    axs[1][0].scatter(D_12['P(active)_1'], D_12['bb2'], color='blue')
    axs[1][2].scatter(D_32['P(active)_3'], D_32['bb2'], color='green')
    axs[1][0].set_xticks([])
    axs[1][1].set_xticks([])
    axs[1][2].set_xticks([])
    p2_ticks = np.arange(0, 120, 20)
    axs[1][0].set_ylim([-5, 105])
    axs[1][0].set_ylabel('compatible BBs in $p_{2}$', fontsize=24, labelpad=18)
    axs[1][0].set_yticks(p2_ticks)
    axs[1][0].set_yticklabels(labels=p2_ticks, fontsize=18)
    axs[1][1].set_yticks([])
    axs[1][2].set_yticks([])

    axs[2][0].scatter(D_13['P(active)_1'], D_13['bb3'], color='blue')
    axs[2][1].scatter(D_23['P(active)_2'], D_23['bb3'], color='orange')
    x_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[2][0].set_xticks(x_ticks)
    axs[2][0].set_xticklabels(labels=x_ticks, fontsize=18)
    axs[2][0].set_xlabel('$p_{1}$ P(active)', fontsize=24, labelpad=18)
    axs[2][1].set_xticks(x_ticks)
    axs[2][1].set_xticklabels(labels=x_ticks, fontsize=18)
    axs[2][1].set_xlabel('$p_{2}$ P(active)', fontsize=24, labelpad=18)
    axs[2][2].set_xticks(x_ticks)
    axs[2][2].set_xticklabels(labels=x_ticks, fontsize=18)
    axs[2][2].set_xlabel('$p_{3}$ P(active)', fontsize=24, labelpad=18)
    p3_ticks = np.arange(0, 900, 100)
    axs[2][0].set_xlim([-0.05, 1.05])
    axs[2][0].set_ylim([-20, 820])
    axs[2][0].set_ylabel('compatible BBs in $p_{3}$', fontsize=24, labelpad=18)
    axs[2][0].set_yticks(p3_ticks)
    axs[2][0].set_yticklabels(labels=p3_ticks, fontsize=18)
    axs[2][1].set_yticks([])
    axs[2][2].set_yticks([])

    line_1 = plt.Line2D((0.2, 0.8), (0.5, 0.5), lw=1.5, color='gray')
    axs[0][0].add_line(line_1)

    circle_1 = plt.Circle((0.2, 0.5), 0.1, fc='blue', ec='None', zorder=3)
    axs[0][0].add_patch(circle_1)

    pts_1 = np.array([[0.5-(0.2/np.sqrt(3)), 0.4], [0.5+(0.2/np.sqrt(3)), 0.4], [0.5, 0.6]])
    tri_1 = Polygon(pts_1, closed=True, fc='white', ec='darkorange', linestyle='dashed', zorder=3)
    axs[0][0].add_patch(tri_1)

    rect_1 = plt.Rectangle((0.7,0.4), 0.2, 0.2, fc='white', ec='green', linestyle='dashed', zorder=3)
    axs[0][0].add_patch(rect_1)

    line_2 = plt.Line2D((0.2, 0.8), (0.5, 0.5), lw=1.5, color='gray')
    axs[1][1].add_line(line_2)

    circle_2 = plt.Circle((0.2, 0.5), 0.1, fc='white', ec='blue', linestyle='dashed', zorder=3)
    axs[1][1].add_patch(circle_2)

    pts_2 = np.array([[0.5-(0.2/np.sqrt(3)), 0.4], [0.5+(0.2/np.sqrt(3)), 0.4], [0.5, 0.6]])
    tri_2 = Polygon(pts_2, closed=True, fc='darkorange', ec='None', zorder=3)
    axs[1][1].add_patch(tri_2)

    rect_2 = plt.Rectangle((0.7,0.4), 0.2, 0.2, fc='white', ec='green', linestyle='dashed', zorder=3)
    axs[1][1].add_patch(rect_2)

    line_3 = plt.Line2D((0.2, 0.8), (0.5, 0.5), lw=1.5, color='gray')
    axs[2][2].add_line(line_3)

    circle_3 = plt.Circle((0.2, 0.5), 0.1, fc='white', ec='blue', linestyle='dashed', zorder=3)
    axs[2][2].add_patch(circle_3)

    pts_3 = np.array([[0.5-(0.2/np.sqrt(3)), 0.4], [0.5+(0.2/np.sqrt(3)), 0.4], [0.5, 0.6]])
    tri_3 = Polygon(pts_3, closed=True, fc='white', ec='darkorange', linestyle='dashed', zorder=3)
    axs[2][2].add_patch(tri_3)

    rect_3 = plt.Rectangle((0.7,0.4), 0.2, 0.2, fc='green', ec='None', zorder=3)
    axs[2][2].add_patch(rect_3)
    
    return D_12, D_13, D_21, D_23, D_31, D_32
    
def apply_bins(bb_pactive, bb_pos):
    all_bins = np.arange(0, 1.13, 0.13)
    bb_pactive[f'{bb_pos}_bins'] = pd.cut(bb_pactive['P(active)'], all_bins, labels=np.arange(len(all_bins)-1)).fillna(0)
    return bb_pactive

def plot_2D_bins(total_complete, bb1_pactive, bb2_pactive, bb3_pactive):      
    ab = total_complete.groupby(['bb1_bins', 'bb2_bins'], as_index=False)['active'].mean()
    val_1_2 = np.zeros((8,8))
    for ind_1, ind_2, active_rate in zip(ab['bb1_bins'], ab['bb2_bins'], ab['active']):
        val_1_2[7-ind_2, ind_1] = active_rate

    bc = total_complete.groupby(['bb2_bins', 'bb3_bins'], as_index=False)['active'].mean()
    val_2_3 = np.zeros((8,8))
    for ind_2, ind_3, active_rate in zip(bc['bb2_bins'], bc['bb3_bins'], bc['active']):
        val_2_3[7-ind_3, ind_2] = active_rate

    ac = total_complete.groupby(['bb1_bins', 'bb3_bins'], as_index=False)['active'].mean()
    val_1_3 = np.zeros((8,8))
    for ind_1, ind_3, active_rate in zip(ac['bb1_bins'], ac['bb3_bins'], ac['active']):
        val_1_3[7-ind_3, ind_1] = active_rate
    
    fig, axs = plt.subplots(1, 3, figsize=(20,6), dpi=150)
    plt.subplots_adjust(wspace=0.7)

    axes_fs=13
    labels_fs=15
    axlabel_fs=20
    labelpad=20
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])
    all_bins = np.arange(0, 1.13, 0.13)

    a, a_ticks, a_labels = set_bins(bb1_pactive['P(active)'], bins=all_bins)
    b, b_ticks, b_labels = set_bins(bb2_pactive['P(active)'], bins=all_bins)
    c, c_ticks, c_labels = set_bins(bb3_pactive['P(active)'], bins=all_bins)

    sns.heatmap(val_1_2[-4:, :4], cmap='viridis', annot=False, ax=axs[0], fmt='.2g',annot_kws={'fontsize': labels_fs},
                cbar_ax=cbar_ax, linewidth=0.01, linecolor='black')

    axs[0].set_xticklabels(a_labels[:4], rotation=20, ha='right', fontsize=axes_fs)
    axs[0].set_yticklabels(b_labels[3::-1], rotation=0, ha='right', fontsize=axes_fs)
    axs[0].set_xlabel('$p_1$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)
    axs[0].set_ylabel('$p_2$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)


    sns.heatmap(val_1_3[:, :4], cmap='viridis', annot=False, ax=axs[1], fmt='.2g',annot_kws={'fontsize': labels_fs}, 
                cbar_ax=cbar_ax, linewidth=0.01, linecolor='black')

    axs[1].set_xticklabels(a_labels[:4], rotation=20, ha='right', fontsize=axes_fs)
    axs[1].set_yticklabels(c_labels[::-1], rotation=0, ha='right', fontsize=axes_fs)
    axs[1].set_xlabel('$p_1$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)
    axs[1].set_ylabel('$p_3$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)


    sns.heatmap(val_2_3[:, :4], cmap='viridis', annot=False, ax=axs[2], fmt='.2g',annot_kws={'fontsize': labels_fs},
                cbar_ax=cbar_ax, linewidth=0.5, linecolor='black')
    axs[2].set_xticklabels(b_labels[:4], rotation=20, ha='right', fontsize=axes_fs)
    axs[2].set_yticklabels(c_labels[::-1], rotation=0, ha='right', fontsize=axes_fs)
    axs[2].set_xlabel('$p_2$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)
    axs[2].set_ylabel('$p_3$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)

    cbar_ax.set_ylabel('probability', fontsize=axlabel_fs, labelpad=labelpad)
    cbar_ax.tick_params(axis='y', labelsize=labels_fs)
    
    return val_1_2[-4:, :4], val_1_3[:, :4], val_2_3[:, :4]

def dist_mat(sim_mat):
    dist_mat = np.max(sim_mat) - ( (sim_mat + sim_mat.T)/2 )
    np.fill_diagonal(dist_mat, 0)
    return dist_mat

def umap_transform(dist_mat):
    U = umap.UMAP(random_state=0, metric='precomputed')
    transform = U.fit(dist_mat)
    return transform

def get_dist_bins(bb_pactive, trans_bb, bb_dist, bb_pos):
    inds = np.triu_indices(len(bb_pactive), k=1)
    bb_dist_umap = distance_matrix(trans_bb.embedding_, trans_bb.embedding_)
    N = len(np.unique(bb_pactive[f'{bb_pos}_bins']))
    dist_arr = np.zeros((N+1, 2))
    for i in range(N):
        bb_bin = list(bb_pactive.loc[bb_pactive[f'{bb_pos}_bins'] == i].index)
        bin_inds = np.triu_indices(len(bb_bin), k=1)
        dist_arr[i, 0] = np.mean(bb_dist[bb_bin, :][:, bb_bin][bin_inds])
        dist_arr[i, 1] = np.mean(bb_dist_umap[bb_bin, :][:, bb_bin][bin_inds])
        
    dist_arr[-1, 0] = np.mean(bb_dist[inds])
    dist_arr[-1, 1] = np.mean(bb_dist_umap[inds])
    return dist_arr

def normalize_range(OldMin, OldMax, NewMin, NewMax, OldValue):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def plot_umap(bb1_pactive, bb2_pactive, bb3_pactive, trans_bb1, trans_bb2, trans_bb3):
    A = bb1_pactive.copy(deep=True)
    B = bb2_pactive.copy(deep=True)
    C = bb3_pactive.copy(deep=True)
    bb1_size = [normalize_range(0, np.max(A['P(active)']), 1, 50, x) for x in A['P(active)']]
    bb2_size = [normalize_range(0, np.max(B['P(active)']), 1, 50, x) for x in B['P(active)']]
    bb3_size = [normalize_range(0, np.max(C['P(active)']), 1, 50, x) for x in C['P(active)']]

    bb1_alpha = [normalize_range(0, np.max(A['P(active)']), 0.1, 1, x) for x in A['P(active)']]
    bb2_alpha = [normalize_range(0, np.max(B['P(active)']), 0.3, 1, x) for x in B['P(active)']]
    bb3_alpha = [normalize_range(0, np.max(C['P(active)']), 0.1, 1, x) for x in C['P(active)']]

    bb1_colors = [[0.0, 0.0, 1, x] for x in bb1_alpha]
    bb2_colors = [[1.0, 0.549, 0, x] for x in bb2_alpha]
    bb3_colors = [[0.0, 0.50196, 0, x] for x in bb3_alpha]

    fig, axs = plt.subplots(2, 3, dpi=150, figsize=(20,10), 
                            gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    
    A[['X','Y']] = trans_bb1.embedding_
    B[['X','Y']] = trans_bb2.embedding_
    C[['X','Y']] = trans_bb3.embedding_

    axs[0][0].scatter(A['X'], A['Y'], s=bb1_size, color=bb1_colors)
    axs[0][0].set_xlabel('UMAP_1', fontsize=14, labelpad=10)
    axs[0][0].set_ylabel('UMAP_2', fontsize=14, labelpad=10)
    axs[0][0].tick_params(axis='both', labelsize=12)

    axs[0][1].scatter(B['X'], B['Y'], s=bb2_size, color=bb2_colors)
    axs[0][1].set_xlabel('UMAP_1', fontsize=14, labelpad=10)
    axs[0][1].set_ylabel('UMAP_2', fontsize=14, labelpad=10)
    axs[0][1].tick_params(axis='both', labelsize=12)

    axs[0][2].scatter(C['X'], C['Y'], s=bb3_size, color=bb3_colors)
    axs[0][2].set_xlabel('UMAP_1', fontsize=14, labelpad=10)
    axs[0][2].set_ylabel('UMAP_2', fontsize=14, labelpad=10)
    axs[0][2].tick_params(axis='both', labelsize=12)
    
    bb1_dist_mat = distance_matrix(trans_bb1.embedding_, trans_bb1.embedding_)
    bb2_dist_mat = distance_matrix(trans_bb2.embedding_, trans_bb2.embedding_)
    bb3_dist_mat = distance_matrix(trans_bb3.embedding_, trans_bb3.embedding_)
    
    top_ind = A.sort_values(by='P(active)', ascending=False).head(10).index
    rand_ind = A.sample(n=10, random_state=42).index
    bb1_top = bb1_dist_mat[top_ind, :][:, top_ind]
    bb1_rand = bb1_dist_mat[rand_ind, :][:, rand_ind]
    bb1_top_rand = bb1_dist_mat[top_ind, :][:, rand_ind]
    
    top_ind = B.sort_values(by='P(active)', ascending=False).head(10).index
    rand_ind = B.sample(n=10, random_state=42).index
    bb2_top = bb2_dist_mat[top_ind, :][:, top_ind]
    bb2_rand = bb2_dist_mat[rand_ind, :][:, rand_ind]
    bb2_top_rand = bb2_dist_mat[top_ind, :][:, rand_ind]
    
    top_ind = C.sort_values(by='P(active)', ascending=False).head(10).index
    rand_ind = C.sample(n=10, random_state=42).index
    bb3_top = bb3_dist_mat[top_ind, :][:, top_ind]
    bb3_rand = bb3_dist_mat[rand_ind, :][:, rand_ind]
    bb3_top_rand = bb3_dist_mat[top_ind, :][:, rand_ind]

    sns.kdeplot(bb1_top[np.triu_indices(10, k=1)], color='blue', ax=axs[1][0], linewidth=2)
    sns.kdeplot(bb1_top_rand.ravel(), color='blue', linestyle='dotted', ax=axs[1][0], linewidth=3)
    axs[1][0].set_xlim(left=0)
    axs[1][0].tick_params(axis='both', which='major', labelsize=14)
    axs[1][0].set_ylabel('Density', labelpad=15, fontsize=20)

    sns.kdeplot(bb2_top[np.triu_indices(10, k=1)], color='darkorange', ax=axs[1][1], linewidth=2)
    sns.kdeplot(bb2_top_rand.ravel(), color='darkorange', linestyle='dotted', ax=axs[1][1], linewidth=3)

    axs[1][1].set_xlim(left=0)
    axs[1][1].set_ylabel('')
    axs[1][1].tick_params(axis='both', which='major', labelsize=14)

    axs[1][1].set_xlabel('Distance', labelpad=15, fontsize=20)

    sns.kdeplot(bb3_top[np.triu_indices(10, k=1)], color='green', ax=axs[1][2], linewidth=2)
    sns.kdeplot(bb3_top_rand.ravel(), color='green', linestyle='dotted', ax=axs[1][2], linewidth=3)

    axs[1][2].set_xlim(left=0)
    axs[1][2].set_ylabel('')
    axs[1][2].tick_params(axis='both', which='major', labelsize=14)

    df = pd.DataFrame([[1, np.mean(bb1_top[np.triu_indices(10, k=1)]), np.mean(bb1_top_rand.ravel())],
              [2, np.mean(bb2_top[np.triu_indices(10, k=1)]), np.mean(bb2_top_rand.ravel())],
              [3, np.mean(bb3_top[np.triu_indices(10, k=1)]), np.mean(bb3_top_rand.ravel())]], 
              columns=['Position', 'top - top dist', 'top - rand dist'])

    display(df)
    return fig, A, B, C

def intracluster_dist(df):
    dist_mat = pairwise_distances(df[['X', 'Y']])
    #dist_mat = pairwise_distances(coord)
    N = len(dist_mat)
    ind = np.triu_indices(N, k=1)
    return np.mean(dist_mat[ind])

def obj(params):
    return params['n_noise'] + 10*params['icd']

def hdbscan_param_search(df, transform):
    BB = df.copy(deep=True)
    BB[['X','Y']] = transform.embedding_
    hdb_params = pd.DataFrame(columns=['min_cluster_size', 'min_samples', 'n_noise', 'icd'])
    for i in np.arange(3,61):
        for j in np.arange(1,21):
            if i < j:
                pass
            else:
                info = {}
                coords = BB[['X','Y']]
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
                
    return hdb_params

def optimal_params(hdb_params):
    ind = np.argmin(obj(hdb_params))
    opt = hdb_params.iloc[ind]
    return (int(opt['min_cluster_size']), int(opt['min_samples']))

def find_optimal_hdbscan(X, min_cluster_size=np.arange(3,61), min_samples=np.arange(1,11), method='silhouette'):
    # initialize the best score and parameters
    best_score = -100
    best_params = (0, 0)

    # iterate over a range of possible min_cluster_sizes and min_samples
    for i in min_cluster_size:
        for j in min_samples:
            if j > i:
                pass
            else:
                # cluster the data using HDBSCAN with the current parameters
                clusterer = hdbscan.HDBSCAN(min_cluster_size=int(i), min_samples=int(j))
                clusters = clusterer.fit_predict(X)
                
                if method == 'silhouette':
                    score = silhouette_score(X, clusters, metric='euclidean') #higher is better [-1, 1]
                elif method == 'calinski_harabasz_score':
                    score = calinski_harabasz_score(X, clusters) # higher is better
                elif method == 'davies_bouldin_score':
                    score = -1*davies_bouldin_score(X, clusters) #lower is better, 0

                # update the best score and parameters if the current score is better
                if score > best_score:
                    #print(f"Best parameters: min_cluster_size={i}, min_samples={j}, score={score:.2f}")
                    best_score = score
                    best_params = (int(i), int(j))

    # return the best parameters
    return best_params


def set_colors(cluster_labels):
    if np.sum(cluster_labels == -1) > 0:
        color = plt.cm.rainbow(np.linspace(0, 1, len(set(cluster_labels))-1))
        colors = np.vstack([color, [0.86, 0.86, 0.86, 1]])
        return colors
    else:
        color = plt.cm.rainbow(np.linspace(0, 1, len(set(cluster_labels))))
        return color
    
def plot_hdbscan(bb_pactive, params, transform, bb_test=None):
    fig, axs = plt.subplots(figsize=(7,7))
    cluster = hdbscan.HDBSCAN(min_cluster_size=params[0], min_samples=params[1], metric='euclidean', gen_min_span_tree=True, 
                          allow_single_cluster=False, prediction_data=True).fit(transform.embedding_)

    bb_pactive['Cluster'] = cluster.labels_
    cluster_colors = set_colors(cluster.labels_)
    axs.scatter(transform.embedding_[:, 0], transform.embedding_[:, 1], color=cluster_colors[cluster.labels_], s=10)

    axs.set_title(f'Number of Clusters: {len(np.unique(cluster.labels_))-1}\nNoise points: {np.unique(cluster.labels_, return_counts=True)[1][0]}')
    if isinstance(bb_test, type(None)) == False:
        axs.scatter(bb_test['X'], bb_test['Y'], color=cluster_colors[bb_test['Cluster']], s=50, edgecolors='black')

    plt.show()
    return bb_pactive

def plot_hdbscan_interactive(bb_pactive, params, transform, bb_pos, bb_test=None):
    fig, axs = plt.subplots(figsize=(10,10))
    cluster = hdbscan.HDBSCAN(min_cluster_size=params[0], min_samples=params[1], metric='euclidean', gen_min_span_tree=True, 
                          allow_single_cluster=False, prediction_data=True).fit(transform.embedding_)

    bb_pactive['Cluster'] = cluster.labels_
    cluster_colors = set_colors(cluster.labels_)
    bb_pactive['Molecule'] = [smiles_to_oemol(x) for x in bb_pactive[bb_pos]]
    molImgs_hits = list(bb_pactive['Molecule'].apply(lambda x: oenb.draw_mol_to_img_tag(x, 300,200)))

    axs.set_title(f'Number of Clusters: {len(np.unique(cluster.labels_))-1}\nNoise points: {np.unique(cluster.labels_, return_counts=True)[1][0]}')
    if isinstance(bb_test, type(None)) == False:
        axs.scatter(bb_test['X'], bb_test['Y'], color=cluster_colors[bb_test['Cluster']], s=50, edgecolors='black')

    sc = axs.scatter(transform.embedding_[:, 0], transform.embedding_[:, 1], color=cluster_colors[cluster.labels_], s=10)
    
    tooltip = mpld3.plugins.PointHTMLTooltip(sc, molImgs_hits)
    mpld3.plugins.connect(fig, tooltip)
    return mpld3.display(fig=fig)

def get_ticks(n_bb):
    N = 10
    i = 1
    while len(np.arange(0.5, n_bb+0.5, i)) > N:
        i += 1
    ticks = np.arange(0.5, n_bb+0.5, i)
    labels = np.arange(0, n_bb, i)
    return ticks, labels

def plot_cluster_combos(total_compounds, bb1_pactive, bb2_pactive, bb3_pactive, trans_bb1, trans_bb2, trans_bb3):
    clustered_bb1 = bb1_pactive.loc[bb1_pactive['Cluster'] > -1]
    clustered_bb1 = clustered_bb1.rename(columns={'Cluster': 'bb1_Cluster'})
    clustered_bb2 = bb2_pactive.loc[bb2_pactive['Cluster'] > -1]
    clustered_bb2 = clustered_bb2.rename(columns={'Cluster': 'bb2_Cluster'})
    clustered_bb3 = bb3_pactive.loc[bb3_pactive['Cluster'] > -1]
    clustered_bb3 = clustered_bb3.rename(columns={'Cluster': 'bb3_Cluster'})

    total_merged = total_compounds.merge(clustered_bb1, on='bb1')\
        .rename(columns={'Cluster': 'bb1_Cluster', 'P(active)': 'P(active)_1'})\
        .merge(clustered_bb2, on='bb2')\
        .rename(columns={'Cluster': 'bb2_Cluster', 'P(active)': 'P(active)_2'})\
        .merge(clustered_bb3, on='bb3')\
        .rename(columns={'Cluster': 'bb3_Cluster', 'P(active)': 'P(active)_3'})

    ab = total_merged.groupby(['bb1_Cluster', 'bb2_Cluster'], as_index=False)['active'].mean()
    top_ab = ab.sort_values(by='active', ascending=False).head(10)

    bc = total_merged.groupby(['bb2_Cluster', 'bb3_Cluster'], as_index=False)['active'].mean()
    top_bc = bc.sort_values(by='active', ascending=False).head(10)

    ac = total_merged.groupby(['bb1_Cluster', 'bb3_Cluster'], as_index=False)['active'].mean()
    top_ac = ac.sort_values(by='active', ascending=False).head(10)

    n_bb1 = len(np.unique(bb1_pactive['Cluster']))-1
    n_bb2 = len(np.unique(bb2_pactive['Cluster']))-1
    n_bb3 = len(np.unique(bb3_pactive['Cluster']))-1
    ab_mat = np.ones((n_bb1, n_bb2))*-0.05
    for b1, b2, val in zip(ab['bb1_Cluster'], ab['bb2_Cluster'], ab['active']):
        ab_mat[b1, b2] = val

    ac_mat = np.ones((n_bb1, n_bb3))*-0.05
    for b1, b3, val in zip(ac['bb1_Cluster'], ac['bb3_Cluster'], ac['active']):
        ac_mat[b1, b3] = val

    bc_mat = np.ones((n_bb2, n_bb3))*-0.05
    for b2, b3, val in zip(bc['bb2_Cluster'], bc['bb3_Cluster'], bc['active']):
        bc_mat[b2, b3] = val

    bb1_size = [normalize_range(0, np.max(bb1_pactive['P(active)']), 1, 50, x) for x in bb1_pactive['P(active)']]
    bb2_size = [normalize_range(0, np.max(bb2_pactive['P(active)']), 1, 50, x) for x in bb2_pactive['P(active)']]
    bb3_size = [normalize_range(0, np.max(bb3_pactive['P(active)']), 1, 50, x) for x in bb3_pactive['P(active)']]

    bb1_colors = np.array(set_colors(bb1_pactive['Cluster']))
    bb2_colors = np.array(set_colors(bb2_pactive['Cluster']))
    bb3_colors = np.array(set_colors(bb3_pactive['Cluster']))

    sns.reset_orig()
    fig, axs = plt.subplots(2, 3, dpi=150, figsize=(20,11), 
                           gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(wspace=0.2)

    for i in range(np.max(bb1_pactive['Cluster'])+1):
        sub = bb1_pactive[bb1_pactive['Cluster'] == i]
        xtext, ytext = np.median(sub['X']), np.median(sub['Y'])
        txt = axs[0][0].text(xtext+0.2, ytext, str(i), fontsize=12, weight='bold')

    axs[0][0].scatter(trans_bb1.embedding_[:, 0], trans_bb1.embedding_[:, 1], s=bb1_size, color=bb1_colors[bb1_pactive['Cluster']])
    axs[0][0].tick_params(axis="both", labelsize=12)
    axs[0][0].set_xlabel('UMAP_1', fontsize=12, labelpad=10)
    axs[0][0].set_ylabel('UMAP_2', fontsize=12, labelpad=10)

    for i in range(np.max(bb2_pactive['Cluster'])+1):
        sub = bb2_pactive[bb2_pactive['Cluster'] == i]
        xtext, ytext = np.median(sub['X']), np.median(sub['Y'])
        txt = axs[0][1].text(xtext, ytext, str(i), fontsize=13, fontweight='bold')

    axs[0][1].scatter(trans_bb2.embedding_[:, 0], trans_bb2.embedding_[:, 1], s=bb2_size, color=bb2_colors[bb2_pactive['Cluster']])
    axs[0][1].tick_params(axis="both", labelsize=12)
    axs[0][1].set_xlabel('UMAP_1', fontsize=12, labelpad=10)
    axs[0][1].set_ylabel('UMAP_2', fontsize=12, labelpad=5)

    for i in range(np.max(bb3_pactive['Cluster'])+1):
        sub = bb3_pactive[bb3_pactive['Cluster'] == i]
        xtext, ytext = np.median(sub['X']), np.median(sub['Y'])
        txt = axs[0][2].text(xtext+0.3, ytext, str(i), fontsize=13, fontweight='bold')

    axs[0][2].scatter(trans_bb3.embedding_[:, 0], trans_bb3.embedding_[:, 1], s=bb3_size, color=bb3_colors[bb3_pactive['Cluster']])
    axs[0][2].tick_params(axis="both", labelsize=12)
    axs[0][2].set_xlabel('UMAP_1', fontsize=12, labelpad=10)
    axs[0][2].set_ylabel('UMAP_2', fontsize=12, labelpad=5)

    val_max = np.max(np.concatenate([ab['active'], ac['active'], bc['active']]))
    inc = 0.1
    disp_max = (val_max // inc) * inc + inc

    cbar_ax = fig.add_axes([.92, .125, .015, .23])
    bb1_ticks, bb1_labels = get_ticks(n_bb1)
    bb2_ticks, bb2_labels = get_ticks(n_bb2)
    bb3_ticks, bb3_labels = get_ticks(n_bb3)
    ax = sns.heatmap(ab_mat, vmin=-inc, vmax=disp_max, ax=axs[1][0], cmap='viridis', cbar_ax=cbar_ax,)
    axs[1][0].set_xticks(bb2_ticks)
    axs[1][0].set_xticklabels(bb2_labels, fontsize=14)
    axs[1][0].set_yticks(bb1_ticks)
    axs[1][0].set_yticklabels(bb1_labels, fontsize=14, rotation=0)
    axs[1][0].set_xlabel('$p_2$ cluster id', fontsize=18, labelpad=10)
    axs[1][0].set_ylabel('$p_1$ cluster id', fontsize=18, labelpad=10)

    sns.heatmap(ac_mat, vmin=-inc, vmax=disp_max, ax=axs[1][1], cmap='viridis', cbar_ax=cbar_ax)
    axs[1][1].set_xticks(bb3_ticks)
    axs[1][1].set_xticklabels(bb3_labels, fontsize=14, rotation=0)
    axs[1][1].set_yticks(bb1_ticks)
    axs[1][1].set_yticklabels(bb1_labels, fontsize=14, rotation=0)
    axs[1][1].set_xlabel('$p_3$ cluster id', fontsize=18, labelpad=10)
    axs[1][1].set_ylabel('$p_1$ cluster id', fontsize=18, labelpad=10)

    sns.heatmap(bc_mat, vmin=-inc, vmax=disp_max, ax=axs[1][2], cmap='viridis', cbar_ax=cbar_ax)
    axs[1][2].set_xticks(bb3_ticks)
    axs[1][2].set_xticklabels(bb3_labels, fontsize=14, rotation=0)
    axs[1][2].set_yticks(bb2_ticks)
    axs[1][2].set_yticklabels(bb2_labels, fontsize=14, rotation=0)
    axs[1][2].set_xlabel('$p_3$ cluster id', fontsize=18, labelpad=10)
    axs[1][2].set_ylabel('$p_2$ cluster id', fontsize=18, labelpad=10)

    cbar = ax.collections[0].colorbar

    cbar.set_ticks(np.arange(-inc, disp_max+inc, inc))
    labels = [f'{x:.2f}' for x in np.arange(0, disp_max+inc, inc)]
    cbar.set_ticklabels(['no data'] + labels)
    cbar.ax.tick_params(labelsize='x-large')
    cbar.ax.set_ylabel('probability', fontsize=16)
    
    return ab_mat, ac_mat, bc_mat

# Get one representative structure for each cluster
def cluster_rep(bb_pactive, bb_pos):
    BB = bb_pactive.copy(deep=True)
    ranking = BB.groupby(['Cluster'])['P(active)'].rank(method='first', ascending=False)
    num = BB.groupby(['Cluster'], as_index=False).agg(Num = (bb_pos,'nunique'))
    ind = np.where(ranking == 1)[0]
    top = BB.iloc[ind]
    BB_cluster = pd.merge(top, num).rename(columns={bb_pos: 'SMILES'}).sort_values(by='Cluster')
    BB_data = BB.rename(columns={bb_pos: 'SMILES'})
    return BB_cluster, BB_data

def get_largest_fragment(mol):
    '''
    Source: https://gist.github.com/PatWalters/3bb9f1091d989665caf37460fec969f3
    A simple function to return largest fragment in a molecule. Useful for stripping counterions.

    Input
    -----
    mol : RDKit mol
        RDKit molecule object

    Output
    ------
    frags : RDKit mol
        largest fragment from input RDKit mol object
    '''
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    frags.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
    return frags[0]

def display_cluster_members(df, sel, align_mcs=False):
    '''
    Source: https://gist.github.com/PatWalters/3bb9f1091d989665caf37460fec969f3
    A function to generate an image of molecules in a selected cluster.

    Input
    -----
    df : dataframe
        cluster information for selected cluster
    sel : selection
        selection when working with mol2grid graphical interface
    align_mcs : bool
        set whether to aligned based on maximum common substructure; does not work too well with ring systems
    hdbscan : bool
        indicate whether the results will be fed into HDBSCAN clustering
        if True, prints strength of cluster membership along with P(active) value when visualizing compounds in clusters

    Output
    ------
    img : Image
        returns visualization of compounds for the cluster selected via the mols2grid graphical interface
    '''
    mol_list = []
    img = 'Nothing selected'
    if len(sel):
        sel_df = df.query('Cluster +1  in @sel')
        sel_df = sel_df.sort_values(by='P(active)', ascending=False)
        # order by ascending P(active)
        mol_list = [Chem.MolFromSmiles(smi) for smi in sel_df['SMILES']]
        # strip counterions
        mol_list = [get_largest_fragment(x) for x in mol_list]
        # align structures on the MCS
        if align_mcs and len(mol_list) > 1:
            mcs = rdFMCS.FindMCS(mol_list)
            mcs_query = Chem.MolFromSmarts(mcs.smartsString)
            AllChem.Compute2DCoords(mcs_query)
            for m in mol_list:
                AllChem.GenerateDepictionMatching2DStructure(m, mcs_query)
        legends = [f'P(active): {x:.4f}' for x in sel_df['P(active)']]
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, maxMols=40, legends=legends, useSVG=True)
    return img

def gen_random_cluster(bb_pactive):
    BB = bb_pactive.copy(deep=True)
    cluster_id, cluster_freq = np.unique(BB['Cluster'], return_counts=True)
    compounds = list(BB.index)
    random_ids = np.zeros(len(compounds))
    for val, count in zip(cluster_id, cluster_freq):
        assign_id = np.random.choice(compounds, size=count, replace=False)
        random_ids[assign_id] = val
        compounds = list(set(compounds) - set(assign_id))
    BB['Cluster'] = [int(x) for x in random_ids]
    #expand = pd.melt(BB, id_vars=[bb_pos, 'P(active)'], value_vars=['Cluster', 'rand_Cluster'],
    #                 var_name='method', value_name='cluster_id')
    return BB

def FWHM(X,Y):
    # taken from: https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    half_max = np.max(Y) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #find the left and right most indexes
    left_idx = min(np.where(d > 0)[0])
    right_idx = max(np.where(d < 0)[-1])
    return float(X[right_idx] - X[left_idx]) #return the difference (full width)

def calc_fwhm(bb_pactive):
    N_clusters = len(np.unique(bb_pactive['Cluster'])) - 1
    bb_fwhm = {}
    for i in range(N_clusters):
        kde = sns.kdeplot(bb_pactive.loc[bb_pactive['Cluster'] == i, 'P(active)'])
        if len(kde.lines) > 0:
            line = kde.lines[0]
            x,y = line.get_data()
            bb_fwhm[i] = FWHM(x,y)
        else:
            bb_fwhm[i] = 0
        plt.figure()
    plt.close("all")
    return bb_fwhm

## Calculate FWHM for 100 random trials
def fwhm_ttest(bb_active, bb_fwhm, N):
    n_clust = len(np.unique(bb_active['Cluster'])) - 1
    fwhm_arr = np.zeros((N, n_clust))
    for n in range(N):
        bb_random = gen_random_cluster(bb_active)
        fwhm_arr[n, :] = list(calc_fwhm(bb_random).values())

    act_fwhm = bb_fwhm['FWHM'].to_numpy()
    rand_fwhm = fwhm_arr.mean(axis=0)
    bb_fwhm_rand = pd.DataFrame(rand_fwhm, columns=['FWHM']).assign(random=True)
    return ttest_ind(act_fwhm, rand_fwhm, equal_var=True), bb_fwhm_rand

def plot_cluster_pactive(bb_pactive, bb_random):
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    subfigs = fig.subfigures(1, 2, wspace=0, hspace=0.01)

    N = np.sum(np.unique(bb_pactive['Cluster']) != -1)
    axsLeft = subfigs[0].subplots(N, 1, sharex=True)
    plt.subplots_adjust(hspace=-0.6)
    bb_colors = np.array(set_colors(bb_pactive['Cluster']))
    for index, ax in enumerate(axsLeft):
        ax.set_xlim([10**-5, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.patch.set_alpha(0.01)
        ax.spines[['left','right', 'top']].set_visible(False)
        subset = bb_pactive.loc[(bb_pactive['Cluster'] == index) & (bb_pactive['P(active)'] > 0)]
        if index != N-1:
            ax.tick_params(axis='x', which='both', bottom=False)
            sns.kdeplot(subset['P(active)'], ax=ax, color=bb_colors[index], fill=True, log_scale=True)
        else:
            sns.kdeplot(subset['P(active)'], ax=ax, color=bb_colors[index], fill=True, log_scale=True)
            ax.set_xlabel(xlabel='P(active)', fontsize=18, labelpad=10)
            ax.set_xticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1])
            ax.set_xticklabels(labels=['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=14)

    axsRight = subfigs[1].subplots(N, 1, sharex=True)
    subfigs[1].suptitle('Random Cluster')
    for index, ax in enumerate(axsRight):
        ax.set_xlim([10**-5, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.minorticks_off()
        ax.patch.set_alpha(0.01)
        ax.spines[['left','right', 'top']].set_visible(False)
        subset = bb_random.loc[(bb_random['Cluster'] == index) & (bb_random['P(active)'] > 0)]
        if index != N-1:
            ax.tick_params(axis='x', which='both', bottom=False)
            sns.kdeplot(subset['P(active)'], ax=ax, color=bb_colors[index], fill=True, log_scale=True)
        else:
            sns.kdeplot(subset['P(active)'], ax=ax, color=bb_colors[index], fill=True, log_scale=True)
            ax.set_xlabel(xlabel='P(active)', fontsize=18, labelpad=10)
            ax.set_xticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1])
            ax.set_xticklabels(labels=['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=14)


