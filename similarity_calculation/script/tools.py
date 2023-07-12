import numpy as np
from openeye import oechem, oeomega, oegraphsim
import pickle

def read_error(logfile, indfile):
    '''
    Verify that conformers were properly generated for all molecules needing enumerated stereochemistry
    '''
    ## Parse error file to get indices of trouble molecules
    problem_ind = []
    with open(logfile) as fp:
        for index, line in enumerate(fp):
            if (index - 1) % 3 == 0:
                problem_ind.append(line.split(' ')[2])

    problem_ind = set([int(x) for x in problem_ind])

    ## Load in grouping dict and verify all trouble molecules had stereoisomers enumerated
    grouping = pickle.load(open(indfile, 'rb'))
    grouping_ind = set(grouping.keys())

    if problem_ind == grouping_ind:
        print('Conformers successfully generated for all molecules!')
        return True
    else:
        print('Check molecules {}'.format(problem_ind ^ grouping_ind))
        return False

def smiles_to_oemol(smiles, title='MOL'):
    molecule = oechem.OEMol()
    if not oechem.OEParseSmiles(molecule, smiles):
        raise ValueError("The supplied SMILES '%s' could not be parsed." % smiles)
    molecule.SetTitle(title)
    return molecule

def normalize_molecule(molecule):
    molcopy = oechem.OEMol(molecule)
    oechem.OEAssignAromaticFlags(molcopy, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(molcopy)
    if any([atom.GetName() == '' for atom in molcopy.GetAtoms()]):
        oechem.OETriposAtomNames(molcopy)
    return molcopy

def generate_conformers(molecule, max_confs=200):
    """Generate conformations for the supplied molecule
    Parameters
    ----------
    ###Keyword arguments have been altered to default to same as vanilla OMEGA###
    molecule : OEMol
        Molecule for which to generate conformers
    max_confs : int, optional, default=800
        Max number of conformers to generate.  If None, use default OE Value.
    strictStereo : bool, optional, default=True
        If False, permits smiles strings with unspecified stereochemistry.
    strictTypes : bool, optional, default=True
        If True, requires that Omega have exact MMFF types for atoms in molecule; otherwise, allows the closest atom type of the same element to be used.
    Returns
    -------
    molcopy : OEMol
        A multi-conformer molecule with up to max_confs conformers.
    Notes
    -----
    Roughly follows
    http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """

    molcopy = oechem.OEMol(molecule)
    omega = oeomega.OEOmega()

    ## Explicitly list out all settings
    ## These correspond to the defaults set in the Orion implementation of OMEGA
    omega.SetIncludeInput(False)
    omega.SetCanonOrder(True)
    omega.SetSampleHydrogens(False)
    omega.SetEnergyWindow(5.0)
    omega.SetRMSThreshold(0.5)
    omega.SetStrictAtomTypes(False)

    ## This setting is False in Orion, but we set to True in order to catch
    ## compounds with unspecified stereocenters.
    ## We enumerate the stereoisomers in a separate step
    omega.SetStrictStereo(True)

    if max_confs is not None:
        omega.SetMaxConfs(max_confs)

    # Generate Conformation
    status = omega(molcopy)

    if not status:
        raise(RuntimeError("omega returned error code %d" % status))

    return molcopy

def write_dataframe_to_file(df, filename, mol_col="Molecule", title_col=None, clear_data=True):
    """
    Taken from OpenEye oenotebook package: https://anaconda.org/openeye/openeye-oenotebook

    Write a `DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ containing
    molecules out to a molecule file.

    :param df: Input dataframe to write
    :param filename: Filename in which to save molecules. Must be a valid format for
        `OEChem <http://docs.eyesopen.com/toolkits/python/oechemtk/molreadwrite.html>`_
    :param mol_col: Column name where the molecules are stored in the dataframe
    :param title_col: Column name of Titles to be set on molecules before writing
    :param clear_data: If True, all SD tags will be cleared from molecules before adding in those in the dataframe
        columns. If False, tags that correspond to columns are updated, but other tags are not removed.
    """
    ofs = oechem.oemolostream()

    if not ofs.open(filename):
        oechem.OEThrow.Fatal("Unable to open {} for writing".format(filename))

    for i in df.index:
        mol = oechem.OEMol(df.loc[i][mol_col])
        if clear_data:
            oechem.OEClearSDData(mol)
        for c in df.columns:
            if c == mol_col:
                continue
            if c == title_col:
                mol.SetTitle(df.loc[i][title_col])

            oechem.OEAddSDData(mol,c,str(df.loc[i][c]))

        oechem.OEWriteMolecule(ofs, mol)

def read_file_to_dataframe(filename, mol_col="Molecule", title_col=None, tag_iterator_limit=10):
    """
    Taken from OpenEye oenotebook package: https://anaconda.org/openeye/openeye-oenotebook

    Read a molecule file into a `pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_,
    making columns for the SD data.

    :param filename: Filename from which to read molecules. Must be a valid format for
        `OEChem <http://docs.eyesopen.com/toolkits/python/oechemtk/molreadwrite.html>`_
    :param mol_col: Column name where the molecules will be stored in the dataframe
    :param title_col: If provided, the column name for the Titles of the molecules
    :param tag_iterator_limit: number of molecules to parse and look for SD tags
    :return: returns a dataframe containing molecules and SD data
    """
    import pandas as pd
    ifs = oechem.oemolistream()

    if not ifs.open(filename):
        raise IOError

    tagSet = set()
    idxCounter = 0
    df = pd.DataFrame(columns=[mol_col])
    for mol in ifs.GetOEMols():
        df.loc[idxCounter] = [oechem.OEMol(mol)]

        if idxCounter < tag_iterator_limit:
            for sddata in oechem.OEGetSDDataIter(mol):
                tagSet.add(sddata.GetTag())

        idxCounter += 1

    lookOnConfs = False
    # If we don't find any tags, look on first conformer
    if len(tagSet) == 0:
        for i in range(min(tag_iterator_limit,df[mol_col].count())):
            for sddata in oechem.OEGetSDDataIter(df[mol_col].iloc[i].GetActive()):
                tagSet.add(sddata.GetTag())

        if len(tagSet) > 0:
            lookOnConfs = True

    if title_col is not None:
        df[title_col] = df[mol_col].apply(lambda x: x.GetTitle())

    for tag in tagSet:
        if lookOnConfs:
            df[tag] = df[mol_col].apply(lambda x: _convert_str(oechem.OEGetSDData(x.GetActive(), tag)) if oechem.OEHasSDData(x.GetActive(), tag) else None)
        else:
            df[tag] = df[mol_col].apply(lambda x: _convert_str(oechem.OEGetSDData(x, tag)) if oechem.OEHasSDData(x, tag) else None)

    return df

def _convert_str(val):
    try:
        if "." in val:
            val = float(val)
        else:
            val = int(val)
    except ValueError:
        pass
    return val

def gen_fp(smiles, fptype=oegraphsim.OEFPType_Circular):
    fps = []
    for smi in smiles:
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, smi)
        fp = oegraphsim.OEFingerPrint()
        oegraphsim.OEMakeFP(fp, mol, fptype)
        fps.append(fp)
    return fps

def fp_matrix(fps):
    N = len(fps)
    t_matrix = np.zeros((N,N), np.float)
    for n in range(N):
        for m in range(n, N):
            t_matrix[n,m] = oegraphsim.OETanimoto(fps[n], fps[m])
    return t_matrix
