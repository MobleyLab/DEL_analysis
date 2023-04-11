import hdbscan
from IPython.display import display, Image
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import mols2grid
import mpld3
import numpy as np
import oenotebook as oenb
from openeye import oechem
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from scipy.spatial import distance_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, pairwise_distances
import umap

def calc_pactive(data, bb_list, bb_pos):
    '''
    Calculate P(active) value for each building block.
    
    Input
    -----
    data : dataframe
        contains SMILES of each composite structure; SMILES of the constituent building blocks; experimental read count
    
    bb_list : dataframe
        building blocks ordered in the same way as what was used to calculate the similarity matrix
        
    bb_pos : str
        string to specify which building block position; can be either "bb1", "bb2" or "bb3"
    
    Output
    ------
    merged : dataframe
        dataframe containing the SMILES of each building block at the specified position and its P(active) value
    '''
    # Label compounds as 'active' or 'inactive' based on experimental read count values
    data['active'] = [0 if x == 0 else 1 for x in data['read_count']]
    # Calculate P(active) for each building block at each position
    val = data.groupby([bb_pos], as_index=False)['active'].mean().rename(columns={'active': 'P(active)'})
    # Merge dataframe to the list of building blocks used to calculate the similarity matrix to maintain consistent indexing
    merged = pd.merge(bb_list, val, left_on='SMILES', right_on=bb_pos, how='left').drop(columns=['SMILES'])
    return merged

def set_bins(bb_pactive, bb_bins):
    '''
    Generate bins to separate building blocks at each position by P(active) value.
    
    Input
    -----
    bb_pactive : dataframe
        dataframe containing the SMILES of each building block at the specified position and its P(active) value
        
    bb_bins : array
        array with the bins to separate building blocks
    
    Output
    ------
    info : array
        contains the bins and the number of building blocks in each bin
    
    bb_ticks : array
        contains position for each axis tick
        
    bb_labels : array
        contains label for each P(active) bin
    '''
    # Set tick to middle of each bin
    bb_ticks = (bb_bins[:-1] + bb_bins[1:])/2
    # Create tick labels
    bb_labels = []
    for i in range(len(bb_bins)-1):
        # Cut-off bin at P(active) = 1 if range extends beyond
        if bb_bins[i+1] >= 1:
            bb_labels.append(f'[{bb_bins[i]:.2f}, 1.00]')
        # Make upper bound inclusive if max ends before P(active) = 1
        elif i+1 == len(bb_bins)-1:
            bb_labels.append(f'[{bb_bins[i]:.2f}, {bb_bins[i+1]:.2f}]')
        # Create bin labels that are inclusive of lower bound but exclusive of upper bound
        else:
            bb_labels.append(f'[{bb_bins[i]:.2f}, {bb_bins[i+1]:.2f})')
    info = np.histogram(bb_pactive, bb_bins)
    return info, bb_ticks, bb_labels

def plot_pactive(bb1_pactive, bb2_pactive, bb3_pactive, all_bins):
    '''
    Creates a figure of the distribution of P(active) values for each building block position.
    
    Input
    -----
    bb1_pactive, bb2_pactive, bb3_pactive : dataframe
        dataframe containing the SMILES and P(active) value of each building block in the corresponding position
        
    all_bins : array
        array specifying the P(active) bins to divide building blocks into
    
    Output
    ------
    a, b, c : array
        building block counts in each P(active) bin for positions 1, 2 and 3 respectively
    '''
    # Create bin assignments, tick locations and tick labels
    a, a_ticks, a_labels = set_bins(bb1_pactive['P(active)'], bb_bins=all_bins)
    b, b_ticks, b_labels = set_bins(bb2_pactive['P(active)'], bb_bins=all_bins)
    c, c_ticks, c_labels = set_bins(bb3_pactive['P(active)'], bb_bins=all_bins)
    
    # Plot spread of P(active) values for building blocks in position 1
    fig, axs = plt.subplots(3, 1, figsize=(9,14), dpi=150, sharey=True)
    plt.subplots_adjust(hspace=0.15)
    a_bars = axs[0].bar(a_ticks, height=a[0], width=0.06, log=True, color='blue')
    axs[0].bar_label(a_bars, fontsize=18, padding=0)
    axs[0].set_xticks([])
    axs[0].set_yticks(np.array([1, 10, 100, 1000, 10000]))
    axs[0].set_yticklabels(np.array([1, 10, 100, 1000, 10000]), fontsize=20)
    axs[0].set_xlim([0, 1])

    # Plot spread of P(active) values for building blocks in position 2
    b_bars = axs[1].bar(b_ticks, height=b[0], width=0.06, log=True, color='orange')
    axs[1].bar_label(b_bars, fontsize=18, padding=0)
    axs[1].set_xticks([])
    axs[1].set_yticks(np.array([1, 10, 100, 1000, 10000]))
    axs[1].set_yticklabels(np.array([1, 10, 100, 1000, 10000]), fontsize=20)
    axs[1].set_xlim([0, 1])
    axs[1].set_ylabel('number of building blocks', fontsize=28, labelpad=20)

    
    # Plot spread of P(active) values for building blocks in position 3
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
    '''
    Displays the top N building blocks in the specified position by descending order of P(active).
    
    Input
    -----
    bb_pactive : dataframe
        dataframe containing the SMILES and P(active) value of each building block
    
    bb_pos : string
        string to specify which building block position; can be either "bb1", "bb2" or "bb3"
        
    N : int
        the number of building blocks to display
    
    Output
    ------
    img : Image
        RDKit rendering of the top N building blocks
    '''
    # Sort building blocks by P(active)
    bb_sorted = bb_pactive.sort_values(by='P(active)', ascending=False)
    bb_top = bb_sorted[:N]
    # Convert SMILES string to molecule object
    bb_mols = [Chem.MolFromSmiles(smi) for smi in bb_top['iso_SMILES']]
    # Draw top N building blocks by P(active) at each position
    img = Draw.MolsToGridImage(bb_mols, molsPerRow=N, returnPNG=False,
                               legends=[f'P(active): {x:.3f}' for x in bb_top['P(active)']])
    return img

def merge_df(df, bb1_pactive, bb2_pactive, bb3_pactive):
    '''
    Appends each compound with the P(active) values of its constituent building blocks.
    
    Input
    -----
    df : dataframe
        dataframe containing all the compounds to be analyzed
     
    bb1_pactive, bb2_pactive, bb3_pactive : dataframe
        dataframe containing the SMILES and P(active) value of each building block in the corresponding position
    
    Output
    ------
    merged : dataframe
        dataframe with all compounds and the P(active) value of their constituent building block
    '''
    # Merge compound list with the P(active) value of the building block at each position
    merged = df.merge(bb1_pactive, on='bb1')\
        .rename(columns={'P(active)': 'P(active)_1'})\
        .merge(bb2_pactive, on='bb2')\
        .rename(columns={'P(active)': 'P(active)_2'})\
        .merge(bb3_pactive, on='bb3')\
        .rename(columns={'P(active)': 'P(active)_3'})
    return merged

def get_actives(total):
    '''
    Returns active compounds from the total list of compounds. 
    
    Input
    -----
    total : dataframe
        dataframe of all compounds to be analyzed
    
    Output
    ------
    actives : dataframe
        dataframe consisting only of compounds with experimental read count greater than 0
    '''
    # Query all compounds that have greater than 0 experimental read count
    total['active'] = [0 if x == 0 else 1 for x in total['read_count']]
    actives = total.loc[total['active'] == 1]
    return actives

def plot_compatible(total_actives):
    '''
    Plots the number of compatible building blocks as a function of building block P(active).
    
    Input
    -----
    total_actives : dataframe
        dataframe of all active compounds

    Output
    ------
    D_12, D_13, D_21, D_23, D_31, D_32 : dataframe
        dataframe with the SMILES of a building block, its P(active) and the number of compatible building blocks it has in a different position 
    '''
    # Calculate number of compatible building blocks in positions 2 and 3 for each building block in position 1
    D_12 = total_actives.groupby(['bb1', 'P(active)_1'], as_index=False)['bb2'].nunique()
    D_13 = total_actives.groupby(['bb1', 'P(active)_1'], as_index=False)['bb3'].nunique()

    # Calculate number of compatible building blocks in positions 1 and 3 for each building block in position 2
    D_21 = total_actives.groupby(['bb2', 'P(active)_2'], as_index=False)['bb1'].nunique()
    D_23 = total_actives.groupby(['bb2', 'P(active)_2'], as_index=False)['bb3'].nunique()

    
    # Calculate number of compatible building blocks in positions 1 and 2 for each building block in position 3
    D_31 = total_actives.groupby(['bb3', 'P(active)_3'], as_index=False)['bb1'].nunique()
    D_32 = total_actives.groupby(['bb3', 'P(active)_3'], as_index=False)['bb2'].nunique()

    fig, axs = plt.subplots(3, 3, figsize=(20, 20), dpi=150, sharex=True)
    plt.subplots_adjust(wspace=0.12, hspace=0.05)

    # Plot first row
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

    # Plot second row
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

    #Plot third row
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
    
    # Format graphics in the main diagonal of the subplots
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
    
def apply_bins(bb_pactive, bb_pos, all_bins):
    '''
    Assigns numerical bin for each building block based on its P(active) value.
    
    Input
    -----
    bb_pactive : dataframe
        dataframe with SMILES of each building block and its P(active) value
        
    bb_pos : string
        string indicating the building block position; can be either "bb1", "bb2" or "bb3"
    
    all_bins : array
        array of the P(active) bins to divide building blocks into

    Output
    ------
    bb_pactive : dataframe
        updated version of bb_pactive with the P(active) bin of each building block
    '''
    bb_pactive[f'{bb_pos}_bins'] = pd.cut(bb_pactive['P(active)'], all_bins, labels=np.arange(len(all_bins)-1)).fillna(0)
    return bb_pactive

def plot_2D_bins(total_complete, bb1_pactive, bb2_pactive, bb3_pactive, all_bins):
    '''
    Returns pairwise heatmaps for P(active) bins at each building block position.
    
    Input
    -----
    total_complete : dataframe
        dataframe of all compounds with the P(active) bin of the constituent building blocks merged on
    
    bb1_pactive, bb2_pactive, bb3_pactive : dataframe
        dataframe containing the SMILES of the building blocks at each position and their P(active) value
    
    all_bins : array
        array of the P(active) bins to divide building blocks into
    
    Output
    ------
    val_1_2, val_1_3, val_2_3 : array
        array of the probability of forming active compounds for combinations of P(active) bins
    '''
    n_bins = len(all_bins)-1
    # Calculate probability of forming active compound for combinations of P(active) bins in positions 1 and 2
    ab = total_complete.groupby(['bb1_bins', 'bb2_bins'], as_index=False)['active'].mean()
    val_1_2 = np.zeros((n_bins,n_bins))
    for ind_1, ind_2, active_rate in zip(ab['bb1_bins'], ab['bb2_bins'], ab['active']):
        val_1_2[n_bins-1-ind_2, ind_1] = active_rate

    # Calculate probability of forming active compound for combinations of P(active) bins in positions 2 and 3
    bc = total_complete.groupby(['bb2_bins', 'bb3_bins'], as_index=False)['active'].mean()
    val_2_3 = np.zeros((n_bins,n_bins))
    for ind_2, ind_3, active_rate in zip(bc['bb2_bins'], bc['bb3_bins'], bc['active']):
        val_2_3[n_bins-1-ind_3, ind_2] = active_rate

    # Calculate probability of forming active compound for combinations of P(active) bins in positions 1 and 3
    ac = total_complete.groupby(['bb1_bins', 'bb3_bins'], as_index=False)['active'].mean()
    val_1_3 = np.zeros((n_bins,n_bins))
    for ind_1, ind_3, active_rate in zip(ac['bb1_bins'], ac['bb3_bins'], ac['active']):
        val_1_3[n_bins-1-ind_3, ind_1] = active_rate

    fig, axs = plt.subplots(1, 3, figsize=(20,6), dpi=150)
    plt.subplots_adjust(wspace=0.7)

    axes_fs=16
    labels_fs=18
    axlabel_fs=20
    labelpad=20
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])

    # Create bin assignments, tick locations and tick labels
    a, a_ticks, a_labels = set_bins(bb1_pactive['P(active)'], bb_bins=all_bins)
    b, b_ticks, b_labels = set_bins(bb2_pactive['P(active)'], bb_bins=all_bins)
    c, c_ticks, c_labels = set_bins(bb3_pactive['P(active)'], bb_bins=all_bins)

    # Get all P(active) bins that have building blocks for each position
    A = np.sum(np.where(a[0] != 0))
    B = np.sum(np.where(b[0] != 0))
    C = np.sum(np.where(c[0] != 0))

    # Plot pairwise probability of forming active compound for building block P(active) in position 1 and 2
    sns.heatmap(val_1_2[-B:, :A], cmap='viridis', annot=False, ax=axs[0], fmt='.2g',annot_kws={'fontsize': labels_fs},
                cbar_ax=cbar_ax, linewidth=0.01, linecolor='black')
    axs[0].set_xticklabels(a_labels[:A], rotation=20, ha='right', fontsize=axes_fs)
    axs[0].set_yticklabels(b_labels[B-1::-1], rotation=0, ha='right', fontsize=axes_fs)
    axs[0].set_xlabel('$p_1$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)
    axs[0].set_ylabel('$p_2$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)

    # Plot pairwise probability of forming active compound for building block P(active) in position 1 and 3
    sns.heatmap(val_1_3[:C, :A], cmap='viridis', annot=False, ax=axs[1], fmt='.2g',annot_kws={'fontsize': labels_fs}, 
                cbar_ax=cbar_ax, linewidth=0.01, linecolor='black')
    axs[1].set_xticklabels(a_labels[:A], rotation=20, ha='right', fontsize=axes_fs)
    axs[1].set_yticklabels(c_labels[C-1::-1], rotation=0, ha='right', fontsize=axes_fs)
    axs[1].set_xlabel('$p_1$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)
    axs[1].set_ylabel('$p_3$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)

    # Plot pairwise probability of forming active compound for building block P(active) in position 2 and 3
    sns.heatmap(val_2_3[:C, :B], cmap='viridis', annot=False, ax=axs[2], fmt='.2g',annot_kws={'fontsize': labels_fs},
                cbar_ax=cbar_ax, linewidth=0.5, linecolor='black')
    axs[2].set_xticklabels(b_labels[:B], rotation=20, ha='right', fontsize=axes_fs)
    axs[2].set_yticklabels(c_labels[C-1::-1], rotation=0, ha='right', fontsize=axes_fs)
    axs[2].set_xlabel('$p_2$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)
    axs[2].set_ylabel('$p_3$ P(active) bin', labelpad=labelpad, fontsize=axlabel_fs)

    cbar = axs[2].collections[0].colorbar
    cbar.set_ticks(np.linspace(0.01, 0.99, 6))
    labels = [f'{x:.2f}' for x in np.arange(0, 1.2, 0.2)]
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize='x-large')
    cbar.ax.set_ylabel('probability', fontsize=labels_fs)
    
    return val_1_2[-B:, :A], val_1_3[:C, :A], val_2_3[:C, :B]

def dist_mat(sim_mat):
    '''
    Converts pairwise similarity matrix into pairwise distance matrix. 
    
    Input
    -----
    sim_mat : array
        array of pairwise similarity values for all the building blocks in a position
    
    Output
    ------
    dist_mat : array
        array of pairwise distances for all the building blocks in a position
    '''
    # Take inverse of similarity to get distances and symmetrize 
    dist_mat = np.max(sim_mat) - ( (sim_mat + sim_mat.T)/2 )
    # Fill main diagonal with 0
    np.fill_diagonal(dist_mat, 0)
    return dist_mat

def umap_transform(dist_mat):
    '''
    Creates a UMAP object to generate 2D coordinates from a pairwise distance matrix.
    
    Input
    -----
    dist_mat : array
        array of pairwise distances for all the building blocks in a position
    
    Output
    ------
    transform : UMAP object
        UMAP object that projects building blocks onto a 2D coordinate space
    '''
    # Transform chemical distances into distances in UMAP space
    U = umap.UMAP(random_state=0, metric='precomputed')
    transform = U.fit(dist_mat)
    return transform

def normalize_range(OldMin, OldMax, NewMin, NewMax, OldValue):
    # Scale values from one range to another
    # Source: StackOverflow 
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def plot_umap(bb1_pactive, bb2_pactive, bb3_pactive, trans_bb1, trans_bb2, trans_bb3):
    '''
    Returns 2D UMAP plot of the building blocks at each position and a density plot of the distance between building blocks in the projected space.
    
    Input
    -----
    bb1_pactive, bb2_pactive, bb3_pactive : dataframe
        dataframe containing SMILES of each building block, its P(active) value and the corresponding P(active) bin
        
    trans_bb1, trans_bb2, trans_bb3 : UMAP object
        UMAP object containing the 2D coordinate projections for each building block position
    
    Output
    ------
    bb1_pactive, bb2_pactive, bb3_pactive : dataframe
        updates each dataframe with the 2D UMAP coordinates of the corresponding building blocks in that position
    '''
    # Rescale marker size based on P(active) value
    bb1_size = [normalize_range(0, np.max(bb1_pactive['P(active)']), 1, 50, x) for x in bb1_pactive['P(active)']]
    bb2_size = [normalize_range(0, np.max(bb2_pactive['P(active)']), 1, 50, x) for x in bb2_pactive['P(active)']]
    bb3_size = [normalize_range(0, np.max(bb3_pactive['P(active)']), 1, 50, x) for x in bb3_pactive['P(active)']]

    # Adjust marker transparency based on P(active) value
    bb1_alpha = [normalize_range(0, np.max(bb1_pactive['P(active)']), 0.1, 1, x) for x in bb1_pactive['P(active)']]
    bb2_alpha = [normalize_range(0, np.max(bb2_pactive['P(active)']), 0.3, 1, x) for x in bb2_pactive['P(active)']]
    bb3_alpha = [normalize_range(0, np.max(bb3_pactive['P(active)']), 0.1, 1, x) for x in bb3_pactive['P(active)']]

    bb1_colors = [[0.0, 0.0, 1, x] for x in bb1_alpha]
    bb2_colors = [[1.0, 0.549, 0, x] for x in bb2_alpha]
    bb3_colors = [[0.0, 0.50196, 0, x] for x in bb3_alpha]

    fig, axs = plt.subplots(2, 3, dpi=150, figsize=(20,10), 
                            gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    
    # Assign X and Y coordinates for each point from UMAP
    bb1_pactive[['X','Y']] = trans_bb1.embedding_
    bb2_pactive[['X','Y']] = trans_bb2.embedding_
    bb3_pactive[['X','Y']] = trans_bb3.embedding_

    # Plot each building block in position 1 in UMAP space scaled by its P(active) value 
    axs[0][0].scatter(bb1_pactive['X'], bb1_pactive['Y'], s=bb1_size, color=bb1_colors)
    axs[0][0].set_xlabel('UMAP_1', fontsize=14, labelpad=10)
    axs[0][0].set_ylabel('UMAP_2', fontsize=14, labelpad=10)
    axs[0][0].tick_params(axis='both', labelsize=12)

    # Plot each building block in position 2 in UMAP space scaled by its P(active) value 
    axs[0][1].scatter(bb2_pactive['X'], bb2_pactive['Y'], s=bb2_size, color=bb2_colors)
    axs[0][1].set_xlabel('UMAP_1', fontsize=14, labelpad=10)
    axs[0][1].set_ylabel('UMAP_2', fontsize=14, labelpad=10)
    axs[0][1].tick_params(axis='both', labelsize=12)

    # Plot each building block in position 3 in UMAP space scaled by its P(active) value 
    axs[0][2].scatter(bb3_pactive['X'], bb3_pactive['Y'], s=bb3_size, color=bb3_colors)
    axs[0][2].set_xlabel('UMAP_1', fontsize=14, labelpad=10)
    axs[0][2].set_ylabel('UMAP_2', fontsize=14, labelpad=10)
    axs[0][2].tick_params(axis='both', labelsize=12)
    
    # Calculate distances between points in UMAP space
    bb1_dist_mat = distance_matrix(trans_bb1.embedding_, trans_bb1.embedding_)
    bb2_dist_mat = distance_matrix(trans_bb2.embedding_, trans_bb2.embedding_)
    bb3_dist_mat = distance_matrix(trans_bb3.embedding_, trans_bb3.embedding_)
    
    # Get distance between top P(active) and randomly selected building blocks in position 1
    top_ind = bb1_pactive.sort_values(by='P(active)', ascending=False).head(10).index
    rand_ind = bb1_pactive.sample(n=10, random_state=42).index
    bb1_top = bb1_dist_mat[top_ind, :][:, top_ind]
    bb1_rand = bb1_dist_mat[rand_ind, :][:, rand_ind]
    bb1_top_rand = bb1_dist_mat[top_ind, :][:, rand_ind]
    
    # Get distance between top P(active) and randomly selected building blocks in position 2
    top_ind = bb2_pactive.sort_values(by='P(active)', ascending=False).head(10).index
    rand_ind = bb2_pactive.sample(n=10, random_state=42).index
    bb2_top = bb2_dist_mat[top_ind, :][:, top_ind]
    bb2_rand = bb2_dist_mat[rand_ind, :][:, rand_ind]
    bb2_top_rand = bb2_dist_mat[top_ind, :][:, rand_ind]
    
    # Get distance between top P(active) and randomly selected building blocks in position 3
    top_ind = bb3_pactive.sort_values(by='P(active)', ascending=False).head(10).index
    rand_ind = bb3_pactive.sample(n=10, random_state=42).index
    bb3_top = bb3_dist_mat[top_ind, :][:, top_ind]
    bb3_rand = bb3_dist_mat[rand_ind, :][:, rand_ind]
    bb3_top_rand = bb3_dist_mat[top_ind, :][:, rand_ind]

    # Plot density of distances between top P(active) building blocks and randomly selected building blocks in position 1
    sns.kdeplot(bb1_top[np.triu_indices(10, k=1)], color='blue', ax=axs[1][0], linewidth=2)
    sns.kdeplot(bb1_top_rand.ravel(), color='blue', linestyle='dotted', ax=axs[1][0], linewidth=3)
    axs[1][0].set_xlim(left=0)
    axs[1][0].tick_params(axis='both', which='major', labelsize=14)
    axs[1][0].set_ylabel('Density', labelpad=15, fontsize=20)

    # Plot density of distances between top P(active) building blocks and randomly selected building blocks in position 2
    sns.kdeplot(bb2_top[np.triu_indices(10, k=1)], color='darkorange', ax=axs[1][1], linewidth=2)
    sns.kdeplot(bb2_top_rand.ravel(), color='darkorange', linestyle='dotted', ax=axs[1][1], linewidth=3)
    axs[1][1].set_xlim(left=0)
    axs[1][1].set_ylabel('')
    axs[1][1].tick_params(axis='both', which='major', labelsize=14)
    axs[1][1].set_xlabel('Distance', labelpad=15, fontsize=20)

    # Plot density of distances between top P(active) building blocks and randomly selected building blocks in position 3
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
    return bb1_pactive, bb2_pactive, bb3_pactive

def intracluster_dist(df):
    '''
    Calculates the average distance between provided points.
    
    Input
    -----
    df : dataframe
        dataframe with building blocks and their X and Y coordinates
    
    Output
    ------
    avg_icd : float
        the mean distance between points
    '''
    dist_mat = pairwise_distances(df[['X', 'Y']])
    N = len(dist_mat)
    ind = np.triu_indices(N, k=1)
    avg_icd = np.mean(dist_mat[ind])
    return avg_icd

def obj(params):
    '''
    Computes the value of objective function for each clustering initialization (see SI for more details)
    
    Input
    -----
    params : dataframe
        dataframe with clustering information for each combination of hyperparameters
    
    Output
    ------
    score : array
        value of the objective function

    '''
    score = params['n_noise'] + 10*params['icd']
    return score

def hdbscan_param_search(df, transform):
    '''
    Performs a hyperparameter search for the HDBSCAN clustering algorithm.
    
    Input
    -----
    df : dataframe
        dataframe containing the SMILES of building blocks, its P(active) value, corresponding P(active) bin and 2D coordinates 
    
    transform : UMAP object
        UMAP object that projects building blocks onto a 2D coordinate space

    Output
    ------
    hdb_params : dataframe
        dataframe with clustering information for each combination of hyperparameters
    '''
    BB = df.copy(deep=True)
    BB[['X','Y']] = transform.embedding_
    hdb_params = pd.DataFrame(columns=['min_cluster_size', 'min_samples', 'n_noise', 'icd'])
    # Loop over range of hyperparameter values for HDBSCAN 
    for i in np.arange(3,61):
        for j in np.arange(1,21):
            if i < j:
                pass
            else:
                info = {}
                coords = BB[['X','Y']]
                # Record clustering information for each HDBSCAN initialization
                cluster = hdbscan.HDBSCAN(min_cluster_size=int(i), min_samples=int(j), metric='euclidean', gen_min_span_tree=True, allow_single_cluster=False).fit(coords)

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
    '''
    Finds hyperparameters corresponding to the minimum of the objective function.

    Input
    -----
    hdb_params : dataframe
        dataframe with clustering information for each combination of hyperparameters
    
    Output
    ------
    opt_params : tuple
        optimal hyperparameters for HDBSCAN based on the minimum of the objective function
    '''
    ind = np.argmin(obj(hdb_params))
    opt = hdb_params.iloc[ind]
    opt_params = ( int(opt['min_cluster_size']), int(opt['min_samples']) )
    return opt_params

def set_colors(cluster_labels):
    '''
    Sets colors to plot points based on their assigned HDBSCAN cluster.
    
    Input
    -----
    cluster_labels : array
        array of cluster IDs
    
    Output
    ------
    color : array
        array of RGB values corresponding to the color of each unique cluster ID
    '''
    # Plot noise points in gray (if they exist)
    if np.sum(cluster_labels == -1) > 0:
        color = plt.cm.rainbow(np.linspace(0, 1, len(set(cluster_labels))-1))
        colors = np.vstack([color, [0.86, 0.86, 0.86, 1]])
        return colors
    # Otherwise, plot points on a color gradient based on their cluster ID
    else:
        color = plt.cm.rainbow(np.linspace(0, 1, len(set(cluster_labels))))
        return color
    
def plot_hdbscan(bb_pactive, params, transform):
    '''
    Plots building blocks in projected UMAP space colored by their HDBSCAN cluster assignment.
    
    Input
    -----
    bb_pactive : dataframe
        contains building block SMILES, P(active) value and 2D UMAP coordinates
    
    params : tuple
        tuple of hyperparameters to initialize HDBSCAN
    
    transform : UMAP object
        UMAP object that projects building blocks onto 2D coordinate space
    
    Output
    ------
    bb_pactive : dataframe
        updates dataframe with HDBSCAN cluster assignment
    '''
    fig, axs = plt.subplots(figsize=(7,7))
    # Generate cluster assignments based on input hyperparameters
    cluster = hdbscan.HDBSCAN(min_cluster_size=params[0], min_samples=params[1], metric='euclidean', gen_min_span_tree=True, 
                          allow_single_cluster=False, prediction_data=True).fit(transform.embedding_)

    bb_pactive['Cluster'] = cluster.labels_
    cluster_colors = set_colors(cluster.labels_)

    # Plot points colored by their HDBSCAN cluster assignment
    axs.scatter(transform.embedding_[:, 0], transform.embedding_[:, 1], color=cluster_colors[cluster.labels_], s=10)
    axs.set_title(f'Number of Clusters: {len(np.unique(cluster.labels_))-1}\nNoise points: {np.unique(cluster.labels_, return_counts=True)[1][0]}')
    plt.show()
    return bb_pactive

def smiles_to_oemol(smi):
    '''
    Convert SMILES string to molecule object.
    
    Input
    -----
    smi : string
        SMILES string of molecule
    
    Output
    ------
    mol : OEMol
        molecule object
    '''
    mol = oechem.OEMol()
    oechem.OEParseSmiles(mol, smi)
    return mol

def plot_hdbscan_interactive(bb_pactive, params, transform, bb_pos):
    '''
    Returns an interactive version of HDBSCAN plot where mousing over points shows image of the corresponding molecule.

    Input
    -----
    bb_pactive : dataframe
        dataframe of building block SMILES, P(active) value, UMAP coordinates and cluster assignment
        
    params : tuple
        tuple of hyperparameters to initialize HDBSCAN
    
    transform : UMAP object
        contains 2D UMAP coordinates of the corresponding building blocks
    
    bb_pos : string
        string specifying which building block position; can be either "bb1", "bb2" or "bb3"
    
    Output
    ------
    interactive figure of building block UMAP projection
    '''
    fig, axs = plt.subplots(figsize=(7,7))
    cluster = hdbscan.HDBSCAN(min_cluster_size=params[0], min_samples=params[1], metric='euclidean', gen_min_span_tree=True, 
                          allow_single_cluster=False, prediction_data=True).fit(transform.embedding_)
    cluster_colors = set_colors(cluster.labels_)

    sc = axs.scatter(transform.embedding_[:, 0], transform.embedding_[:, 1], color=cluster_colors[cluster.labels_], s=10)
    axs.set_title(f'Number of Clusters: {len(np.unique(cluster.labels_))-1}\nNoise points: {np.unique(cluster.labels_, return_counts=True)[1][0]}')
    bb_pactive['Molecule'] = [smiles_to_oemol(x) for x in bb_pactive[bb_pos]]
    molImgs_hits = list(bb_pactive['Molecule'].apply(lambda x: oenb.draw_mol_to_img_tag(x, 300,200)))
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
    '''
    Plots HDBSCAN clusters of UMAP projections of building blocks at each position and pairwise heatmaps for combinations of building block clusters at each position.
    
    Input
    -----
    total_compounds : dataframe
        dataframe of all compounds to be analyzed
        
    bb1_pactive, bb2_pactive, bb3_pactive : dataframe
        dataframe containing the SMILES of building blocks, its P(active) value, P(active) bin, 2D coordinates and cluster assignment

    trans_bb1, trans_bb2, trans_bb3 : UMAP object
        UMAP object containing the 2D coordinate projections for each building block position

    Output
    ------
    ab_mat, ac_mat, bc_mat : array
        array containing the pairwise probability of forming active compounds for combinations of building block clusters in each position

    '''
    # Query building blocks that have cluster assignment for each position (not a noise point)
    clustered_bb1 = bb1_pactive.loc[bb1_pactive['Cluster'] > -1]
    clustered_bb1 = clustered_bb1.rename(columns={'Cluster': 'bb1_Cluster'})
    clustered_bb2 = bb2_pactive.loc[bb2_pactive['Cluster'] > -1]
    clustered_bb2 = clustered_bb2.rename(columns={'Cluster': 'bb2_Cluster'})
    clustered_bb3 = bb3_pactive.loc[bb3_pactive['Cluster'] > -1]
    clustered_bb3 = clustered_bb3.rename(columns={'Cluster': 'bb3_Cluster'})

    # Identify all compounds where the building blocks at all three positions have a cluster assignment
    total_merged = total_compounds.merge(clustered_bb1, on='bb1')\
        .rename(columns={'Cluster': 'bb1_Cluster', 'P(active)': 'P(active)_1'})\
        .merge(clustered_bb2, on='bb2')\
        .rename(columns={'Cluster': 'bb2_Cluster', 'P(active)': 'P(active)_2'})\
        .merge(clustered_bb3, on='bb3')\
        .rename(columns={'Cluster': 'bb3_Cluster', 'P(active)': 'P(active)_3'})

    # Calculate pairwise probability of forming active compounds for building block clusters
    ab = total_merged.groupby(['bb1_Cluster', 'bb2_Cluster'], as_index=False)['active'].mean()
    bc = total_merged.groupby(['bb2_Cluster', 'bb3_Cluster'], as_index=False)['active'].mean()
    ac = total_merged.groupby(['bb1_Cluster', 'bb3_Cluster'], as_index=False)['active'].mean()
    
    n_bb1 = len(np.unique(bb1_pactive['Cluster']))-1
    n_bb2 = len(np.unique(bb2_pactive['Cluster']))-1
    n_bb3 = len(np.unique(bb3_pactive['Cluster']))-1
    
    # Store values in matrix and identify combinations of clusters containing no data
    ab_mat = np.ones((n_bb1, n_bb2))*-0.05
    for b1, b2, val in zip(ab['bb1_Cluster'], ab['bb2_Cluster'], ab['active']):
        ab_mat[b1, b2] = val

    ac_mat = np.ones((n_bb1, n_bb3))*-0.05
    for b1, b3, val in zip(ac['bb1_Cluster'], ac['bb3_Cluster'], ac['active']):
        ac_mat[b1, b3] = val

    bc_mat = np.ones((n_bb2, n_bb3))*-0.05
    for b2, b3, val in zip(bc['bb2_Cluster'], bc['bb3_Cluster'], bc['active']):
        bc_mat[b2, b3] = val

    # Scale marker size by the value of the building block P(active)
    bb1_size = [normalize_range(0, np.max(bb1_pactive['P(active)']), 1, 50, x) for x in bb1_pactive['P(active)']]
    bb2_size = [normalize_range(0, np.max(bb2_pactive['P(active)']), 1, 50, x) for x in bb2_pactive['P(active)']]
    bb3_size = [normalize_range(0, np.max(bb3_pactive['P(active)']), 1, 50, x) for x in bb3_pactive['P(active)']]

    # Color each point by its cluster assignment
    bb1_colors = np.array(set_colors(bb1_pactive['Cluster']))
    bb2_colors = np.array(set_colors(bb2_pactive['Cluster']))
    bb3_colors = np.array(set_colors(bb3_pactive['Cluster']))

    sns.reset_orig()
    fig, axs = plt.subplots(2, 3, dpi=150, figsize=(20,11), 
                           gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(wspace=0.2)

    # Plot clustering in UMAP space for building blocks in position 1
    for i in range(np.max(bb1_pactive['Cluster'])+1):
        sub = bb1_pactive[bb1_pactive['Cluster'] == i]
        xtext, ytext = np.median(sub['X']), np.median(sub['Y'])
        txt = axs[0][0].text(xtext+0.2, ytext, str(i), fontsize=12, weight='bold')

    axs[0][0].scatter(trans_bb1.embedding_[:, 0], trans_bb1.embedding_[:, 1], s=bb1_size, color=bb1_colors[bb1_pactive['Cluster']])
    axs[0][0].tick_params(axis="both", labelsize=12)
    axs[0][0].set_xlabel('UMAP_1', fontsize=12, labelpad=10)
    axs[0][0].set_ylabel('UMAP_2', fontsize=12, labelpad=10)

    # Plot clustering in UMAP space for building blocks in position 2
    for i in range(np.max(bb2_pactive['Cluster'])+1):
        sub = bb2_pactive[bb2_pactive['Cluster'] == i]
        xtext, ytext = np.median(sub['X']), np.median(sub['Y'])
        txt = axs[0][1].text(xtext, ytext, str(i), fontsize=13, fontweight='bold')

    axs[0][1].scatter(trans_bb2.embedding_[:, 0], trans_bb2.embedding_[:, 1], s=bb2_size, color=bb2_colors[bb2_pactive['Cluster']])
    axs[0][1].tick_params(axis="both", labelsize=12)
    axs[0][1].set_xlabel('UMAP_1', fontsize=12, labelpad=10)
    axs[0][1].set_ylabel('UMAP_2', fontsize=12, labelpad=5)

    # Plot clustering in UMAP space for building blocks in position 3
    for i in range(np.max(bb3_pactive['Cluster'])+1):
        sub = bb3_pactive[bb3_pactive['Cluster'] == i]
        xtext, ytext = np.median(sub['X']), np.median(sub['Y'])
        txt = axs[0][2].text(xtext+0.3, ytext, str(i), fontsize=13, fontweight='bold')

    axs[0][2].scatter(trans_bb3.embedding_[:, 0], trans_bb3.embedding_[:, 1], s=bb3_size, color=bb3_colors[bb3_pactive['Cluster']])
    axs[0][2].tick_params(axis="both", labelsize=12)
    axs[0][2].set_xlabel('UMAP_1', fontsize=12, labelpad=10)
    axs[0][2].set_ylabel('UMAP_2', fontsize=12, labelpad=5)

    # Determine spacing for values in heatmap
    val_max = np.max(np.concatenate([ab['active'], ac['active'], bc['active']]))
    inc = 0.1
    disp_max = (val_max // inc) * inc + inc

    cbar_ax = fig.add_axes([.92, .125, .015, .23])
    # Determine increments for axis labels based on number of clusters at each position
    bb1_ticks, bb1_labels = get_ticks(n_bb1)
    bb2_ticks, bb2_labels = get_ticks(n_bb2)
    bb3_ticks, bb3_labels = get_ticks(n_bb3)
    
    # Plot heatmap of cluster combinations in positions 1 and 2
    ax = sns.heatmap(ab_mat, vmin=-inc, vmax=disp_max, ax=axs[1][0], cmap='viridis', cbar_ax=cbar_ax,)
    axs[1][0].set_xticks(bb2_ticks)
    axs[1][0].set_xticklabels(bb2_labels, fontsize=14)
    axs[1][0].set_yticks(bb1_ticks)
    axs[1][0].set_yticklabels(bb1_labels, fontsize=14, rotation=0)
    axs[1][0].set_xlabel('$p_2$ cluster id', fontsize=18, labelpad=10)
    axs[1][0].set_ylabel('$p_1$ cluster id', fontsize=18, labelpad=10)

    # Plot heatmap of cluster combinations in positions 1 and 3
    sns.heatmap(ac_mat, vmin=-inc, vmax=disp_max, ax=axs[1][1], cmap='viridis', cbar_ax=cbar_ax)
    axs[1][1].set_xticks(bb3_ticks)
    axs[1][1].set_xticklabels(bb3_labels, fontsize=14, rotation=0)
    axs[1][1].set_yticks(bb1_ticks)
    axs[1][1].set_yticklabels(bb1_labels, fontsize=14, rotation=0)
    axs[1][1].set_xlabel('$p_3$ cluster id', fontsize=18, labelpad=10)
    axs[1][1].set_ylabel('$p_1$ cluster id', fontsize=18, labelpad=10)

    # Plot heatmap of cluster combinations in positions 2 and 3
    sns.heatmap(bc_mat, vmin=-inc, vmax=disp_max, ax=axs[1][2], cmap='viridis', cbar_ax=cbar_ax)
    axs[1][2].set_xticks(bb3_ticks)
    axs[1][2].set_xticklabels(bb3_labels, fontsize=14, rotation=0)
    axs[1][2].set_yticks(bb2_ticks)
    axs[1][2].set_yticklabels(bb2_labels, fontsize=14, rotation=0)
    axs[1][2].set_xlabel('$p_3$ cluster id', fontsize=18, labelpad=10)
    axs[1][2].set_ylabel('$p_2$ cluster id', fontsize=18, labelpad=10)

    # Format colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(-inc, disp_max+inc, inc))
    labels = [f'{x:.2f}' for x in np.arange(0, disp_max+inc, inc)]
    cbar.set_ticklabels(['no data'] + labels)
    cbar.ax.tick_params(labelsize='x-large')
    cbar.ax.set_ylabel('probability', fontsize=16)
    
    return ab_mat, ac_mat, bc_mat

# Get one representative structure for each cluster
def cluster_rep(bb_pactive, bb_pos):
    '''
    Identifies one representative building block for each cluster.
    
    Input
    -----
    bb_pactive : dataframe
        dataframe with each building block, its P(active) value, UMAP coordinates and cluster assignment
        
    Output
    ------
    '''
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
    Identifies the largest fragment in a molecule. Useful for stripping counterions.

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
    sel : selection object
        selection when working with mol2grid graphical interface
    align_mcs : bool
        set whether to aligned based on maximum common substructure; does not work too well with ring systems

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
        # Order by ascending P(active)
        mol_list = [Chem.MolFromSmiles(smi) for smi in sel_df['SMILES']]
        # Identify largest fragment
        mol_list = [get_largest_fragment(x) for x in mol_list]
        # Align structures on the MCS
        if align_mcs and len(mol_list) > 1:
            mcs = rdFMCS.FindMCS(mol_list)
            mcs_query = Chem.MolFromSmarts(mcs.smartsString)
            AllChem.Compute2DCoords(mcs_query)
            for m in mol_list:
                AllChem.GenerateDepictionMatching2DStructure(m, mcs_query)
        legends = [f'P(active): {x:.4f}' for x in sel_df['P(active)']]
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, maxMols=40, legends=legends, useSVG=True)
    return img

def gen_random_cluster(bb_pactive, bb_pos):
    '''
    Generate clusters consisting of randomly selected building blocks to be used as a control.
    
    Input
    -----
    bb_pactive : dataframe
        dataframe with each building block and its cluster assignment
    bb_pos : str
        string indicating the building block position being evaluated
    
    Output
    ------
    BB : dataframe
        copy of bb_pactive dataframe but with shuffled cluster assignments
    '''
    BB = bb_pactive.copy(deep=True)
    # Get cluster IDs and number of building blocks in each cluster
    cluster_id, cluster_freq = np.unique(BB['Cluster'], return_counts=True)
    compounds = list(BB.index)
    random_ids = np.zeros(len(compounds))
    # Generate clusters of the same size consisting of randomly selected building blocks
    for val, count in zip(cluster_id, cluster_freq):
        assign_id = np.random.choice(compounds, size=count, replace=False)
        random_ids[assign_id] = val
        compounds = list(set(compounds) - set(assign_id))
    BB['Cluster'] = [int(x) for x in random_ids]
    return BB

def plot_cluster_pactive(bb_pactive, bb_random):
    '''
    Plot the distribution of P(active) values for clusters generated with a clustering algorithm and randomly.
    
    Input
    -----
    bb_pactive : dataframe
        dataframe with each building block, its P(active) value and its cluster assignment
    
    bb_random : dataframe
        dataframe with each building block, its P(active) value and a randomly generated cluster assignment
        
    Output
    ------
    '''
    # Initialize figure of figures
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    subfigs = fig.subfigures(1, 2, wspace=0, hspace=0.01)
    N = np.sum(np.unique(bb_pactive['Cluster']) != -1)

    # Plot distribution of P(active) values for each HDBSCAN-generated cluster
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

    # Plot distribution of P(active) values for each randomly-generated cluster
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


