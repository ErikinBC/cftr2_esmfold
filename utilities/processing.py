"""
Utility scripts for the data preprocessing
"""

# Modules
import numpy as np
import pandas as pd
# For multiindex slicing
idx = pd.IndexSlice

# define column metrics of interest
yvars = ['PI','infection','sweat','lung_max_All','lung_min_All','lung_mid_All']

# Define a category mapping
di_category = {'infection':'Infection Rate',
                'sweat':'Sweat Chloride',
                'PI':'Pancreatic Insuff.', 
                'lung_max_All':'FEV1 % (Max)',
                'lung_mid_All':'FEV1 % (Mid)',
                'lung_min_All':'FEV1 % (Min)',
                'lung_min_10':'FEV1% <10 (Min)',
                'lung_min_10-20':'FEV1% 10-20 (Min)',
                'lung_min_20':'FEV1% >20 (Min)',
                'lung_max_10':'FEV1% <10 (Max)',
                'lung_max_10-20':'FEV1% 10-20 (Max)',
                'lung_max_20':'FEV1% >20 (Max)',
                'lung_mid_10':'FEV1% <10 (Mid)',
                'lung_mid_10-20':'FEV1% 10-20 (Mid)',
                'lung_mid_20':'FEV1% >20 (Mid)'
                }
cn_category = list(di_category)
vals_catory = list(di_category.values())

# The y-label type mapping
di_ylbl = {'int':'Mutation average',
           'f508':'F508-hetero',
           'pair':'All-hetero average',
           'homo':'Homozygous'}


def get_y_f508(df:pd.DataFrame) -> pd.DataFrame:
    """
    Processes the y_label.csv to a category-friendly long format labels for F508del mutations
    """
    y = df.loc[idx['F508del'],idx[yvars,'Paired']]
    y = y.droplevel(level=1, axis=1)
    # Melt to include the homozygous combination
    y = y.melt(ignore_index=False).dropna().reset_index()
    y.rename(columns={'mutation2':'mutation'}, inplace=True)
    assert y.groupby(['category','mutation']).size().max() == 1, 'For a given category, we should only have one mutation'
    y.drop(columns=['label_num'], inplace=True)
    assert (y.loc[y['is_homo']=='homo','mutation'] == 'F508del').all(), 'Expected this to be a single value "F508del"'
    return y


def get_y_int(df:pd.DataFrame) -> pd.DataFrame:
    """
    Processes the y_label.csv to a category-friendly long format labels for the average or "integrated" data
    """
    y = df.loc[idx[:,np.nan],idx[:,'Integrated']]
    y = y.droplevel(level=(1,2),axis=0).droplevel(level=(1,2),axis=1)[cn_category]
    y = y.melt(ignore_index=False).dropna().reset_index()
    return y


def get_y_hetero_ave(df:pd.DataFrame) -> pd.DataFrame:
    """
    Processes the y_label.csv to a category-friendly long format labels for the average of all different heterozygous pairs
    """
    y = df.loc[:,idx[cn_category,'Paired']].droplevel(level=1,axis=1)
    y = y[y.notnull().any(1)]
    y = y.melt(ignore_index=False).dropna().groupby(['mutation','category'])['value'].mean().reset_index()
    return y

def get_y_homo(df:pd.DataFrame) -> pd.DataFrame:
    """
    Processes the y_label.csv to a category-friendly long format labels for homozygous combination of the mutation
    """
    y = df.loc[df.index.get_level_values('mutation') == df.index.get_level_values('mutation2'),idx[cn_category,'Paired','homo']]
    y = y.droplevel(level=(1,2),axis=1).droplevel(level=(1,2),axis=0).melt(ignore_index=False).dropna().reset_index()
    return y
