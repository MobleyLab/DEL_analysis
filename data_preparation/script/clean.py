import numpy as np
import pandas as pd
import pickle
from openeye import oechem, oeomega

def update_df(df, drop_ind):
    new_df = df.drop(index=drop_ind)
    return new_df

def find_bad_bbs(df):
    return set(df.loc[df['bb3'] == 'NC1[C@H]2CC3C[C@H]1CC(O)(C3)C2'].index)

def find_null(df):
    null_entries = df[(df['bb1'].str.contains("Null|null"))|(df['bb2'].str.contains("Null|null"))|(df['bb3'].str.contains("Null|null"))]
    return set(null_entries.index)

def find_duplicates(df):
    df['RANK'] = df.groupby(['structure'])['read_count'].rank(method='first', ascending=False)
    return set(df.loc[df['RANK'] != 1.0].index)

def find_nonzero_readcount(df):
    nonzero_rc = df.loc[df['read_count'] != 0]
    return set(nonzero_rc.index)

def find_boron(df):
    comp_1B = set(df.loc[df['bb1'].str.contains('B')].index)
    comp_1Br = set(df.loc[df['bb1'].str.contains('Br')].index)
    remove_1 = comp_1B - comp_1Br

    comp_2B = set(df.loc[df['bb2'].str.contains('B')].index)
    comp_2Br = set(df.loc[df['bb2'].str.contains('Br')].index)
    remove_2 = comp_2B - comp_2Br

    comp_3B = set(df.loc[df['bb3'].str.contains('B')].index)
    comp_3Br = set(df.loc[df['bb3'].str.contains('Br')].index)
    remove_3 = comp_3B - comp_3Br

    return remove_1 | remove_2 | remove_3

def clean_binders(binders):
    binders_drop_1 = find_null(binders)
    binders_drop_2 = find_duplicates(binders)
    binders_drop_3 = find_boron(binders)
    binders_drop_4 = find_bad_bbs(binders)
    binders_updated = update_df(binders, list(binders_drop_1 | binders_drop_2 | binders_drop_3 | binders_drop_4))
    return binders_updated

def clean_nonbinders(nonbinders):
    nonbinders_drop_1 = find_null(nonbinders)
    nonbinders_drop_2 = find_duplicates(nonbinders)
    nonbinders_drop_3 = find_boron(nonbinders)
    nonbinders_drop_4 = find_bad_bbs(nonbinders)
    nonbinders_drop_5 = find_nonzero_readcount(nonbinders)
    nonbinders_updated = update_df(nonbinders, list(nonbinders_drop_1 | nonbinders_drop_2 | nonbinders_drop_3 | nonbinders_drop_4 | nonbinders_drop_5))
    return nonbinders_updated

def get_all_bbs(binders_updated, nonbinders_updated):
    bind_bbs = set(binders_updated['bb1']) | set(binders_updated['bb2']) | set(binders_updated['bb3'])
    nonbind_bbs = set(nonbinders_updated['bb1']) | set(nonbinders_updated['bb2']) | set(nonbinders_updated['bb3'])
    all_bbs = pd.DataFrame({'SMILES': list(bind_bbs | nonbind_bbs)})
    return all_bbs

def has_pg(compound_SMILES, pg):
    '''
    Returns True if the SMILES string contains the protecting group of interest
    '''
    ss = oechem.OESubSearch(pg)
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, compound_SMILES)
    oechem.OEPrepareSearch(mol, ss)
    return ss.SingleMatch(mol)

def deprotectGroup(compound_smi, pg_SMIRKS):
    '''
    Returns the SMILES of the deprotected compound after SMIRKS reaction. If the protecting group is not present
    in the compound, the input SMILES is returned.
    '''
    libgen = oechem.OELibraryGen(pg_SMIRKS)
    libgen.SetValenceCorrection(True)

    ## Rewrite the SMILES to remove kekulization for Fmoc specifically
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, compound_smi)
    rewrite_smi = oechem.OECreateIsoSmiString(mol)

    new_mol = oechem.OEGraphMol()
    oechem.OEParseSmiles(new_mol, rewrite_smi)
    libgen.SetStartingMaterial(new_mol, 0)

    if libgen.NumPossibleProducts() > 0:
        for product in libgen.GetProducts():
            new_smi = oechem.OECreateIsoSmiString(product)

        ## If a different pattern than expected got caught by the query and split
        ## we would prefer to just leave that compound as is
        if '.' in new_smi:
            return rewrite_smi
        else:
            return new_smi

    return rewrite_smi

def return_deprotect(data, subsearch, SMIRKS):
    '''
    Returns a table of all SMILES with an additional column of their deprotected SMILES
    '''
    table = pd.DataFrame(columns=['SMILES', 'nBoc', 'Fmoc', 'ethyl_ester', 'methyl_ester', 'deprot_SMILES'])
    table['SMILES'] = data['SMILES']
    table['deprot_SMILES'] = table['SMILES']

    ## Record whether a given building block has each of the following protecting groups
    table['nBoc'] = table['SMILES'].apply(lambda x: has_pg(x, subsearch['nboc']))
    table['Fmoc'] = table['SMILES'].apply(lambda x: has_pg(x, subsearch['fmoc']))
    table['ethyl_ester'] = table['SMILES'].apply(lambda x: has_pg(x, subsearch['ethyl_ester']))
    table['methyl_ester'] = table['SMILES'].apply(lambda x: has_pg(x, subsearch['methyl_ester']))

    ## Sequentially try to deprotect groups and save finalized structure to the column "deprot_SMILES"
    nboc_deprot = table.loc[table['nBoc']]['deprot_SMILES'].apply(lambda x: deprotectGroup(x, SMIRKS['nboc']))
    table.loc[table['nBoc'], 'deprot_SMILES'] = nboc_deprot

    fmoc_deprot = table.loc[table['Fmoc']]['deprot_SMILES'].apply(lambda y: deprotectGroup(y, SMIRKS['fmoc']))
    table.loc[table['Fmoc'], 'deprot_SMILES'] = fmoc_deprot

    ethyl_deprot = table.loc[table['ethyl_ester']]['deprot_SMILES'].apply(lambda w: deprotectGroup(w, SMIRKS['ethyl_ester']))
    table.loc[table['ethyl_ester'], 'deprot_SMILES'] = ethyl_deprot

    methyl_deprot = table.loc[table['methyl_ester']]['deprot_SMILES'].apply(lambda z: deprotectGroup(z, SMIRKS['methyl_ester']))
    table.loc[table['methyl_ester'], 'deprot_SMILES'] = methyl_deprot
    return table

def deprotect_bbs(all_bbs):
    PG_SMILES = pickle.load(open('../input/PG_SMILES.pkl', 'rb'))
    SMIRKS_patterns = pickle.load(open('../input/deprot_SMIRKS.pkl', 'rb'))
    deprot_bbs = return_deprotect(all_bbs, PG_SMILES, SMIRKS_patterns)
    return deprot_bbs

def isoSMILES(smiles):
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    return oechem.OEMolToSmiles(mol)

def drop_dup(bb_list):
    bb_list['rank'] = bb_list.groupby(['iso_SMILES'], as_index=False)['SMILES'].cumcount()
    return bb_list.loc[bb_list['rank'] == 0].drop(columns='rank')

def repeat_bbs(bb_SMILES):
    '''For when both flat structure and stereoisomer are reported'''
    repeat_dict = {}
    for smi in bb_SMILES:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        for nmol in oeomega.OEFlipper(mol):
            stereo_name = oechem.OEMolToSmiles(nmol)
            has_match = len((np.where(bb_SMILES == stereo_name)[0]))
            if (has_match > 0) & (smi != stereo_name):
                repeat_dict[smi] = stereo_name
    return repeat_dict

def keep_ind(bb_pactive, repeat_bb_dict):
    bb_remove = pd.DataFrame(list(repeat_bb_dict.keys()), columns=['iso_SMILES'])
    itd = pd.merge(bb_pactive, bb_remove, how='left', indicator=True)
    bb_keep = list(itd.loc[itd['_merge'] == 'left_only'].index)
    return bb_keep

def total_concat(binders_updated, nonbinders_updated, deprot_bbs):
    total = pd.concat([binders_updated, nonbinders_updated]).drop(columns=['RANK']).reset_index(drop=True)
    total_deprot = pd.merge(total[['bb1', 'bb2', 'bb3', 'structure', 'read_count']], deprot_bbs[['SMILES', 'deprot_SMILES']], how='left', left_on='bb1', right_on='SMILES')\
        .drop(columns=['SMILES', 'bb1']).rename(columns={'deprot_SMILES': 'bb1'})\
        .merge(deprot_bbs[['SMILES', 'deprot_SMILES']], left_on='bb2', how='left', right_on='SMILES')\
        .drop(columns=['SMILES', 'bb2']).rename(columns={'deprot_SMILES': 'bb2'})\
        .merge(deprot_bbs[['SMILES', 'deprot_SMILES']], left_on='bb3', how='left', right_on='SMILES')\
        .drop(columns=['SMILES', 'bb3']).rename(columns={'deprot_SMILES': 'bb3'})
    return total_deprot

def main():
    # Load in dataframes from experimental runs
    binders = pd.read_csv('../input/del_binders.csv')
    nonbinders = pd.read_csv('../input/del_nonbinders.csv')

    # Clean active and inactive compounds
    cleaned_binders = clean_binders(binders)
    cleaned_nonbinders = clean_nonbinders(nonbinders)

    # Deprotect building blocks in data
    all_bbs = get_all_bbs(cleaned_binders, cleaned_nonbinders)
    deprot_bbs = deprotect_bbs(all_bbs)

    # Organize data into new dataframe
    total_comp = total_concat(cleaned_binders, cleaned_nonbinders, deprot_bbs)

    # Make dataframe of building blocks at each position
    bb1_list = pd.DataFrame({'SMILES': np.unique(total_comp['bb1'])})
    bb2_list = pd.DataFrame({'SMILES': np.unique(total_comp['bb2'])})
    bb3_list = pd.DataFrame({'SMILES': np.unique(total_comp['bb3'])})

    # Generate isomeric SMILES for each building block
    bb1_list['iso_SMILES'] = bb1_list['SMILES'].apply(lambda x: isoSMILES(x))
    bb2_list['iso_SMILES'] = bb2_list['SMILES'].apply(lambda x: isoSMILES(x))
    bb3_list['iso_SMILES'] = bb3_list['SMILES'].apply(lambda x: isoSMILES(x))

    # Identify duplicate building blocks at each position
    bb1_list = drop_dup(bb1_list)
    bb2_list = drop_dup(bb2_list)
    bb3_list = drop_dup(bb3_list)

    # Find building blocks that are duplicated with both unspecified and specified stereochemistry
    repeat_bb1 = repeat_bbs(bb1_list['iso_SMILES'])
    repeat_bb2 = repeat_bbs(bb2_list['iso_SMILES'])
    repeat_bb3 = repeat_bbs(bb3_list['iso_SMILES'])

    # Get index of building blocks to keep
    bb1_keep_ind = keep_ind(bb1_list, repeat_bb1)
    bb2_keep_ind = keep_ind(bb2_list, repeat_bb2)
    bb3_keep_ind = keep_ind(bb3_list, repeat_bb3)

    # Update building blocks at each position
    bb1_new = bb1_list.iloc[bb1_keep_ind].reset_index(drop=True)
    bb2_new = bb2_list.iloc[bb2_keep_ind].reset_index(drop=True)
    bb3_new = bb3_list.iloc[bb3_keep_ind].reset_index(drop=True)

    # Update total compounds
    total_compounds = total_comp.merge(bb1_new, left_on=['bb1'], right_on=['SMILES']).drop(columns=['SMILES']).rename(columns={'iso_SMILES': 'bb1_iso'})\
    .merge(bb2_new, left_on=['bb2'], right_on=['SMILES']).drop(columns=['SMILES']).rename(columns={'iso_SMILES': 'bb2_iso'})\
    .merge(bb3_new, left_on=['bb3'], right_on=['SMILES']).drop(columns=['SMILES']).rename(columns={'iso_SMILES': 'bb3_iso'})

    # Save files
    bb1_new.to_csv('../output/bb1_list.csv', index=False)
    bb2_new.to_csv('../output/bb2_list.csv', index=False)
    bb3_new.to_csv('../output/bb3_list.csv', index=False)
    total_compounds.to_csv('../output/total_compounds.csv', index=False)

if __name__ == "__main__":
    main()
