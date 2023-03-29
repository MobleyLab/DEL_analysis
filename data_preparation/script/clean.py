import numpy as np
import pandas as pd
from openeye import oechem
#from openeye import oechem

def update_df(df, drop_ind):
    return df.drop(index=drop_ind)

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

def clean_hits(hits):
    hits_drop_1 = find_null(hits)
    hits_drop_2 = find_duplicates(hits)
    hits_drop_3 = find_boron(hits)
    hits_drop_4 = find_bad_bbs(hits)
    hits_updated = update_df(hits, list(hits_drop_1 | hits_drop_2 | hits_drop_3 | hits_drop_4))
    return hits_updated

def clean_inactives(inactives):
    inactives_drop_1 = find_null(inactives)
    inactives_drop_2 = find_duplicates(inactives)
    inactives_drop_3 = find_boron(inactives)
    inactives_drop_4 = find_bad_bbs(inactives)
    inactives_drop_5 = find_nonzero_readcount(inactives)
    inactives_updated = update_df(inactives, list(inactives_drop_1 | inactives_drop_2 | inactives_drop_3 | inactives_drop_4 | inactives_drop_5))
    return inactives_updated

def get_all_bbs(hits_updated, inactives_updated):
    hit_bbs = set(hits_updated['bb1']) | set(hits_updated['bb2']) | set(hits_updated['bb3'])
    inactive_bbs = set(inactives_updated['bb1']) | set(inactives_updated['bb2']) | set(inactives_updated['bb3'])
    all_bbs = pd.DataFrame({'SMILES': list(hit_bbs | inactive_bbs)})
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

def output_files(hits_updated, inactives_updated, deprot_bbs):
    total = pd.concat([hits_updated, inactives_updated]).drop(columns=['RANK']).reset_index(drop=True)
    total_deprot = pd.merge(total[['bb1', 'bb2', 'bb3', 'structure', 'read_count']], deprot_bbs[['SMILES', 'deprot_SMILES']], how='left', left_on='bb1', right_on='SMILES')\
        .drop(columns=['SMILES', 'bb1']).rename(columns={'deprot_SMILES': 'bb1'})\
        .merge(deprot_bbs[['SMILES', 'deprot_SMILES']], left_on='bb2', how='left', right_on='SMILES')\
        .drop(columns=['SMILES', 'bb2']).rename(columns={'deprot_SMILES': 'bb2'})\
        .merge(deprot_bbs[['SMILES', 'deprot_SMILES']], left_on='bb3', how='left', right_on='SMILES')\
        .drop(columns=['SMILES', 'bb3']).rename(columns={'deprot_SMILES': 'bb3'})

    bb1_list = pd.DataFrame({'SMILES': np.unique(total_deprot['bb1'])})
    bb2_list = pd.DataFrame({'SMILES': np.unique(total_deprot['bb2'])})
    bb3_list = pd.DataFrame({'SMILES': np.unique(total_deprot['bb3'])})

    total_deprot.to_csv('../output/total_compounds.csv', index=False)
    bb1_list.to_csv('../output/bb1_list.csv', index=False)
    bb2_list.to_csv('../output/bb2_list.csv', index=False)
    bb3_list.to_csv('../output/bb3_list.csv', index=False)
    return True

def main():
    hits = pd.read_csv('../input/del_hits.csv')
    inactives = pd.read_csv('../input/del_inactives.csv')

    cleaned_hits = clean_hits(hits)
    cleaned_inactives = clean_inactives(inactives)

    all_bbs = get_all_bbs(cleaned_hits, cleaned_inactives)

    deprot_bbs = deprotect_bbs(all_bbs)
    output_files(cleaned_hits, cleaned_inactives, deprot_bbs)

if __name__ == "__main__":
    main()
