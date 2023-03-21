"""
Utility scripts
"""

import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from difflib import get_close_matches


def try2df(x) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x


def find_closest_match(x:str, y:list, delta:float=0.05) -> tuple[str, int]:
    """
    Find the closest match in a list (or Series) y to a string x. Will do a reverse search of tolerance - delta*iteration under an apporximate match is found.

    Returns
    -------
    A tuple where the first position is the best match, and the second is the cutoff-tolerance used
    """
    res = []
    a_seq = np.linspace(0, 1, int(1/delta)+1)[::-1]
    i = 0
    while len(res) == 0:
        a_seq_i = a_seq[i]
        tmp = get_close_matches(x, y, n=1, cutoff=a_seq_i)
        if len(tmp) == 1:
            res.append(tmp[0])
        i += 1
    res.append(a_seq_i)
    return res



def force_identical_dfs(df1:pd.DataFrame, df2: pd.DataFrame):
    """
    Force inputs to be dataframes, and then check that they are identical
    """
    df1, df2 = try2df(df1), try2df(df2)
    assert df1.shape == df2.shape
    assert (df1.columns == df2.columns).all()
    assert not df1.index.duplicated().any()
    assert not df2.index.duplicated().any()
    return df1, df2


def merge_pairwise_dfs(df1:pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    If df1 and df2 are identical 
    """
    df1, df2 = force_identical_dfs(df1, df2)
    res = df1.rename_axis('idx').melt(ignore_index=False).reset_index().merge(df2.rename_axis('idx').melt(ignore_index=False).reset_index(),on=['idx','variable'])
    res.dropna(inplace=True)
    res.rename(columns={'variable':'cn'}, inplace=True)
    return res


def cvec(x:np.ndarray) -> np.ndarray:
    """Convert x into a column vector"""
    if len(x.shape) <= 1:
        return x.reshape([len(x),1])
    else:
        return x


def rvec(x:np.ndarray) -> np.ndarray:
    """Convert x into a row vector"""
    if len(x.shape) <= 1:
        return x.reshape([1, len(x)])
    else:
        return x



def no_diff(x, y) -> bool:
    """Check that two strings/lists/Series/arrays are the same same (returns True if so)"""
    x, y = str2list(x), str2list(y)
    check1 = len(np.setdiff1d(x, y)) == 0
    check2 = len(np.setdiff1d(y, x)) == 0
    check = check1 & check2
    return check


def cat_from_map(x:pd.Series, di:dict) -> pd.Series:
    """Convenience wrapper to apply a map and assign a categorical order based on the keys of a dictionary"""
    # Assign categories
    z = pd.Categorical(x, list(di))
    # Apply map
    z = z.map(di)
    # Remove unused categories
    z = z.remove_unused_categories()
    return z


def atleast_4d(x:np.ndarray) -> np.ndarray:
    """Ensure that the input is at least 4D"""
    if x.ndim == 1:
        x = x[None,None,None,:]
    elif x.ndim == 2:
        x = x[None,None,:,:]
    elif x.ndim == 3:
        x = x[None,:,:,:]
    return x



def return_unique_cols(df:pd.DataFrame) -> pd.DataFrame:
    """
    Return the unique columns in a dataframe
    """
    assert isinstance(df, pd.DataFrame), 'df must be a pandas dataframe'
    cn_drop = df.nunique() <= 1
    cn_drop = cn_drop[cn_drop].index
    return df.drop(cn_drop, axis=1)


def str2list(x:str or list) -> list:
    """
    If x is a string, convert to a list
    """
    if isinstance(x, str):
        return [x]
    else:
        return x


def merge_frame_with_existing(df:pd.DataFrame, path:str) -> pd.DataFrame:
    """
    Will attempt to merge a dataframe with an existing dataframe written to local

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to merge
    path : str
        Path to the existing dataframe
    
    Returns
    -------
    pd.DataFrame
        Merged dataframe
    """
    assert isinstance(df, pd.DataFrame), 'df must be a pandas dataframe'
    assert isinstance(path, str), 'path must be a string'
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df]).drop_duplicates().reset_index(drop=True)
    return df



def makeifnot(dir_path):
    """
    Make a directory if it does not exist
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)    


def extract_n(columns:pd.Index) -> np.ndarray:
    """Convenience function to extract the number of patients from the column headers"""
    n_pat = pd.Series(columns[[1,2]].to_list()).str.split('\\(|\\)',regex=True,expand=True,n=2)[1].str.split('\\=\\s',regex=True,expand=True,n=1)[1].str.replace(',','',regex=False).astype(int).values
    return n_pat


def process_2x2(table:pd.DataFrame) -> pd.DataFrame:
    """Process the 2x2 table to extract the value and age"""
    assert isinstance(table, pd.DataFrame)
    # The second and third column of the first row correspond to the mutant specific and general average value
    # The third row and second column corresponds to the average age
    values = table.iloc[0,[1,2]].values
    ages = table.iloc[2,[1,2]].values
    msrs = ['mutant', 'average']
    # Extract the number of patients from the column holders
    n_pat = extract_n(table.columns)
    # Combine and return
    res = pd.DataFrame({'msr':msrs, 'n_pat':n_pat, 'value':values, 'age':ages})
    return res


def process_4x6(table:pd.DataFrame)  -> pd.DataFrame:
    """Process the 2x2 table to extract the value and age"""
    assert isinstance(table, pd.DataFrame)
    # The table is weirdly formatted: rows 3-5 correspond to the mutate specific catgory
    cols = ['age', 'value', 'n_pat']
    tbl1 = pd.DataFrame(table.loc[[2,3,4]].values, columns=cols)
    tbl1.insert(0, 'msr', 'mutant')
    # And rows 6-8 correspond to the general average catgory
    tbl2 = pd.DataFrame(table.loc[[6,7,8]].values, columns=cols)
    tbl2.insert(0, 'msr', 'average')
    # Combine the two tables
    res = pd.concat([tbl1, tbl2], axis=0)
    return res


def merge_html_tables(txt:str) -> pd.DataFrame:
    """
    Wrapper to extract the four different table tables from the html (see column headers and associated processors)
    """
    # Holder to store the different dataframes
    holder = []
    # Find all HTML tables
    tbls = BeautifulSoup(txt, features="lxml").findAll('table')
    for tbl in tbls:
        z = pd.read_html(str(tbl))[0]
        z0 = z.columns[0]
        # If z0 is not the column headers, then skip
        if z0 in column_headers:
            di_z = column_headers[z0]
            category, processor = di_z['tbl'], di_z['process']
            res_z = processor(z)
            res_z.insert(0, 'category', category)
            holder.append(res_z)
    if len(holder) == 0:
        res = pd.DataFrame()
    else:
        res = pd.concat(holder, axis=0)
    return res


# Table headers which need to match the match the first column result currently seen in the "Sweat Chloride", "Lung Function", '% with Pancreatic Insufficiency', 'Pseudomonas Infection Rate'
column_headers = {'Average in people who do not have CF':{'tbl':'sweat', 'process':process_2x2},
    'Lung function range in people who do not have CF':{'tbl':'lung', 'process':process_4x6},
    'Percentage of people without CF who we expect to be pancreatic insufficient':{'tbl':'PI', 'process':process_2x2},
    'Percentage of people without CF who we expect to be infected':{'tbl':'infection', 'process':process_2x2}}


def translate_dna(seq:str) -> str:
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }
    protein = ""
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        if len(codon) == 3:
            protein += table[codon]
    return protein


def get_embedding_moments(di:dict) -> dict:
    """
    For each key in the dictionary (an embedding from the ESMFold model), get the mean, max, min, and standard deviation
    """
    di_pad = {k: atleast_4d(v) for k,v in di.items()}
    di_res = {'mean': {k: np.mean(v,axis=(0,1,2)) for k,v in di_pad.items()},
                'max': {k: np.max(v,axis=(0,1,2)) for k,v in di_pad.items()},
                'min': {k: np.min(v,axis=(0,1,2)) for k,v in di_pad.items()},
                'std': {k: np.std(v.reshape(-1,v.shape[-1]) / v.max(), axis=0) for k,v in di_pad.items()}}    
    return di_res


def diff_btw_matrics(di1:dict, di2:dict, const:int=1000) -> pd.DataFrame:
    """
    Parameters
    ----------
    di1:            Dictionary with three matrices
    di2:            See above
    const:          Normalizing constant for matrix multiplication

    Returns
    -------
    The mean/max/min difference in latent distance for each of the amino acids
    """
    # Input checks
    assert isinstance(di1, dict), 'di1 needs to be a dict'
    assert isinstance(di2, dict), 'di2 needs to be a dict'
    assert no_diff(di1.keys(), di2.keys()), 'Keys between di1/di2 should be the same'
    assert no_diff(list(di1.keys()), ['states','s_s','s_z']), 'Expected one of three key types'
    holder = []
    for k in di1.keys():
        print(f'---- Key = {k} ----')
        mat1 = np.squeeze(di1[k]).copy()
        mat2 = np.squeeze(di2[k].copy())
        mat1.shape;mat2.shape
        if k == 'states':
            mat1, mat2 = mat1.mean(0), mat2.mean(0)
        if k == 's_z':
            mat1 = np.diagonal(mat1, axis1=0, axis2=1).T
            mat2 = np.diagonal(mat2, axis1=0, axis2=1).T
        # Should be a matrix now
        assert len(mat1.shape) == 2
        # Normalize matrices by constant
        mat1 = mat1 / const
        mat2 = mat2 / const
        # Calculate cosine similarity
        num = mat1.dot(mat2.T)
        den1 = np.sqrt(np.square(mat1).sum(1))
        den1 = den1.reshape([len(den1),1])
        den2 = np.sqrt(np.square(mat2).sum(1))
        den2 = den2.reshape([1, len(den2)])
        den = den1 * den2
        cosine = num / den
        # Get the different moments
        data = np.vstack((cosine.mean(axis=1), cosine.max(axis=1), cosine.min(axis=1), cosine.std(axis=1)))
        data = pd.DataFrame(data,index=['mu','mx','mi','se'])
        data = data.melt(ignore_index=False,var_name='amino').rename_axis('moment').reset_index()
        data['amino'] += 1
        data = data.assign(xlbl=lambda x: k.replace('_','')+'_'+x['amino'].astype(str)+'_'+x['moment'])
        data['xlbl'] = pd.Categorical(data['xlbl'],data.sort_values(['moment','amino'])['xlbl'])
        data = data.assign(idx=0).pivot('idx','xlbl','value')
        holder.append(data)
    # Merge and pivot
    res = pd.concat(holder,axis=1)
    return res



def embedding_to_df(di:dict) -> pd.DataFrame:
    """
    Convert the dictionary of embeddings to a dataframe
    """
    # Vertically concatenate the dataframes
    df = pd.concat([pd.concat([pd.DataFrame(v2).assign(msr=k2.replace('_','')) for k2,v2 in v1.items()]).assign(moment=k1) for k1, v1 in di.items()])
    # Rename the value (feature) column
    df.rename(columns={0:'value'}, inplace=True)
    # Get the feature-wise coordinates
    df['feature'] = (df.groupby(['moment','msr']).cumcount()+1).astype(str)
    # Wide-cast to wide-format
    df = df.assign(idx=0).pivot('idx',['msr','moment','feature'],'value')
    df.columns = df.columns.map('_'.join)
    df.reset_index(drop=True, inplace=True)
    return df



def get_cDNA_variant_types(x:pd.Series) -> pd.Series:
    """
    Function to determine the variant type from the cDNA name

    Changes should be one of the following: i) mutation, ii) del, iii) ins, iv) dup, v) inv
    """
    di_variant_types = {'del':'delete', 'ins':'ins', 'dup':'dup', 'inv':'inv', '>':'mutation'}
    z = x.copy().str
    w = z[:0].copy()
    for k, v in di_variant_types.items():
        w.loc[z.contains(k,regex=False)] = v
    w.replace('',np.nan, inplace=True)
    return w



def process_lung_range(df:pd.DataFrame) -> pd.DataFrame:
    """
    Function to process the "lung" category values which are put in a range
    """
    # Split the datafraames
    idx_lung = df['category'] == 'lung'
    tmp1, tmp2 = df[idx_lung], df[~idx_lung]
    # Calculate the min, max, and mid-point
    tmp1b = tmp1['value'].str.split('\\s-\\s',n=1,expand=True,regex=True)
    tmp1b[1] = tmp1b[1].str.replace('\\%','',regex=True)
    tmp1 = pd.concat([tmp1.drop(columns='value'),tmp1b.astype(float).rename(columns={0:'value_min',1:'value_max'})],axis=1)
    tmp1 = tmp1.assign(value_mid=lambda x: (x['value_max'] + x['value_min'])/2)
    # Create an "all" ages
    di_all_ages = {'n_pat':'sum', 'value_min':'min', 'value_max':'max', 'value_mid':'mean'}
    # determine grouping by category
    cn_gg_1c = list(np.setdiff1d(tmp1.columns, list(di_all_ages)+['age']))
    tmp1c = tmp1.groupby(cn_gg_1c).agg(di_all_ages)
    tmp1c = tmp1c.reset_index().assign(age='All')
    tmp1 = pd.concat(objs=[tmp1, tmp1c], axis=0)
    # put into long-format to re-merge
    tmp1 = tmp1.melt(id_vars=np.setdiff1d(tmp2.columns,['value_min','value_max','value_mid','value']), value_vars=['value_min','value_max','value_mid'], var_name='value_type', value_name='value')
    # adjust category name
    tmp1 = tmp1.assign(category=lambda x: 'lung_'+x['value_type'].str.replace('value_','',regex=False))
    tmp1.drop(columns='value_type', inplace=True)
    # ensure values in tmp2 are floats
    tmp2 = tmp2.assign(value=lambda x: x['value'].str.replace('[^0-9\\.]','',regex=True).astype(float))
    # merge and ensure values are floats
    df_ret = pd.concat([tmp1,tmp2],axis=0)
    df_ret['value'] = df_ret['value'].astype(str).astype(float)
    return df_ret


def process_cftr2_mutant(df:pd.DataFrame, cn_gg:str or list=[]) -> pd.DataFrame:
    """
    Processes the CFTR clinical data. Assumes we are looking at the rows of msr=="mutant".
    
    Expects the following columns:
    - msr: The measurement (one of "mutant" or "average")
    - category: The category of clinical measure (one of "PI", "infection", "sweat" or "lung")
    - age: The age
    - n_pat: The number of patients
    - value: The value

    Parameters
    ----------
    df : pd.DataFrame
        The clinical data
    cn_gg : str or list, optional
        The columns to group by, by default []
    """
    # Input checks
    cn_req = ['msr','category','age','n_pat','value']
    assert np.all([c in df.columns for c in cn_req]), f"Missing columns: {cn_req}"
    if len(cn_gg) == 0:
        cn_gg = 'group'
        df = df.copy().assign(group=0)
    # Ensure cn_gg is a list
    cn_gg = str2list(cn_gg)
    # Subset to mutant-rows
    df = df.loc[df['msr'] == 'mutant', cn_gg+cn_req]
    df.drop(columns=['msr'], inplace=True)
    # Clean up missing or irregularly formatted data
    df['n_pat'] = df['n_pat'].fillna(0).astype(int)
    df['value'] = df['value'].str.replace('\\%','',regex=True)
    df.replace('insufficient data',np.nan, inplace=True)
    # Clean up the lung category
    df = process_lung_range(df)
    return df

def get_tril_df(df:pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper to get back the lower triangular part of a square dataframe"""
    assert df.shape[0] == df.shape[1], 'df should be square'
    res = df.stack()[np.triu(np.ones(df.shape)).reshape(df.size) == 0]
    return res

def get_tril_corr(df:pd.DataFrame, method:str or list) -> pd.DataFrame:
    """Convenience wrapper for getting the lower-triangular part of the correlation matrix of a pandas dataframe"""
    if isinstance(method, list):
        holder = []
        for meth in method:
            holder.append(get_tril_corr(df, meth).assign(method=meth))
        return pd.concat(holder).reset_index(drop=True)
    else:
        rho = get_tril_df(df.corr(method=method))
    # Ensure we have unique index names
    rho.index.names = pd.Series(list(rho.index.names)) + pd.Series(['1','2'])
    # Reset and rename column to who
    rho = rho.reset_index().rename(columns={0:'rho'})
    return rho
    

def bootstrap_rho(df:pd.DataFrame, nboot:int=250, alpha:float=0.05, method:str='pearson') -> pd.DataFrame:
    """
    Function to calculate the 100*(1-alpha% CI for the correlation coefficient

    Parameters
    ----------
    df:         A pandas dataframe which the .corr() method can be called on
    nboot:      Number of bootstrap iterations (default=250)
    alpha:      Type-I error rate (default=0.05)
    method:     Valid method to be passed into DataFrame.corr(method='...'), default="pearson"
    """
    assert isinstance(df, pd.DataFrame), 'df should be a dataframe'
    rho0 = get_tril_corr(df, method)
    # Run the bootstrap simulation
    holder_bs = []
    for i in range(nboot):
        res_bs = get_tril_corr(df.sample(frac=1,replace=True,random_state=i),method).assign(idx=i)
        holder_bs.append(res_bs)
    # Merge and aggregate
    cn_gg = list(rho0.columns[:2])
    res_bs = pd.concat(holder_bs).groupby(cn_gg)['rho'].apply(lambda x: pd.DataFrame({'lb':x.quantile(alpha/2), 'ub':x.quantile(1-alpha/2), 'se':x.std(ddof=1)},index=[0])).reset_index()
    res_bs.drop(columns=f'level_{len(cn_gg)}', inplace=True)
    # Add back on point estimate
    rho0 = rho0.merge(res_bs)
    return rho0


def find_arrow_adjustments(df:pd.DataFrame, cn_gg:str or list, cn_x:str, cn_y:str) -> pd.DataFrame:
    """
    For a DataFrame to be fed into a geom_text object, organize the data so that we can plot the data in a way that minimizes overvap
    """
    assert df.columns.isin([cn_x, cn_y]).sum() == 2, 'Cannot find cn_x and/or cn_y'
    cn_gg = str2list(cn_gg)
    assert df.columns.isin(cn_gg).sum() == len(cn_gg), 'Cannot find cn_gg in df'
    df = df.rename(columns={cn_x:'x',cn_y:'y'})
    # Sort by the x-values
    df = df.sort_values(cn_gg+['x','y']).reset_index(drop=True)
    # Find the min/max range ofor the x/y values
    tmp = df.groupby(cn_gg)[['x','y']].agg(['min','max'])
    tmp.columns = tmp.columns.map('_'.join)
    tmp.reset_index(inplace=True)
    df = df.merge(tmp)
    # Determine the linear apportionment
    df = df.assign(idx=lambda x: x.groupby(cn_gg).cumcount())
    df = df.merge(df.groupby(cn_gg)['idx'].agg(['max']).reset_index()).assign(idx=lambda x: x['idx']/x['max']).drop(columns='max')
    df = df.assign(xdelta=lambda x: x['x_max']-x['x_min'], ydelta=lambda x: x['y_max']-x['y_min']).drop(columns=['x_max','y_max'])
    df = df.assign(xlbl=lambda z: z['x_min']+z['xdelta']*z['idx'],
                   ylbl=lambda z: z['y_min']+z['ydelta']*z['idx'])
    # Keep relevant columns
    df.drop(columns=['x_min','xdelta','y_min','ydelta'],inplace=True)
    return df
