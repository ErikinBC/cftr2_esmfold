"""
This script generates the label (y) and feature (X) that will be used for modelling and summary statistics. 

The label output (data/y_label.csv) is stored to have have a MultiIndex for its columns and index.
* The column MultiIndex has three different levels. The first level ("category") has 15 different unique values: [PI, infection, sweat, lung_{moment}_{age}] where moment={min, max, min} and age={<10,10-20,20+,All}, hence the lung combinations have 12 different value types. The second level ("is_pair"), indicates whether it is for a single allele ("Integrated" == cftr_uni.csv) or a pair of alleles ("Paired"==cftr_comb.csv). Note that for the paired alleles, each pair will have two copies so that an identical value will be founded for (mutation, mutation2) and (mutation2, mutation). The third level ("is_homo") indicates the value for a "Paired" mutation when mutation==mutation2.
* The row MultiIndex has three different levels. The first level "mutation" will always have a have, and the second level "mutation2" will have a value when there is a paired observation available, otherwise it will be a np.nan value. The third level ("label_num") gives a count for an index number relative to mutation (e.g. groupby(mutation,mutation2).cumcount()+1).

The script generates two types of fixed-dimension feature output. Each mutation has one of three embeddings with the following dimensions: {'states': (8, 1, n_amino, 384), 's_s': (1, n_amino, 1024), 's_z': (1, n_amino, n_amino, 128)}. 
* The first approach treats the last axis as the index to calculate moments for, and gets the mean, max, min, and std for each of them. For example, the 'states' embedding would a total of 384*4=1536 dimensions. This gets saved as the (data/mutant_embeddings.csv) file with 6144 parameters.
* The second approach calculated the cosine similarity between the "base" class and the mutant class. To reshape each embedding to a fixed vector length, the 'states' embedding is averaged over its first axis to get a (n_amino,384) matrix, and the 's_z' embedding takes the diagonal of its matrix to get a (n_amino, 128) matrix. A cosine similarity matrix is calculate that is of dimension (1480,n_amino), where the (i,j)'th position is the cosine similarity between the base CFTR's embedding at the i'th amino acid, and the mutant's CFTR embedding at the j'th amino acid. Note that 1480 is the number of amino acids of the wildtype CFTR gene (aka "base"). The mean, min, max, and std of this matrix is taken so that for each embedding there are 1480*4=5920 total features. This gets saved as the (data/mutant_cosine.csv) with 5920*3=17760 parameters.

-----------
Note that the default argument for reference_file=="base" comes from the hard-coded name of the first dictonary key of di_polypeptide from the 4_cftr_gene.py script
"""

# Load modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
from mizani.formatters import percent_format
# Load utilities
from parameters import dir_data, dir_esmfold, dir_figures, reference_file
from utilities.utils import get_embedding_moments, embedding_to_df, process_cftr2_mutant, process_lung_range, diff_btw_matrics


#########################
# --- (1) LOAD DATA --- #

# (i) Determine number of ESMFold embeddings
fn_esmfold = pd.Series(os.listdir(dir_esmfold), dtype=str)
mutants_esmfold = fn_esmfold.str.replace('.npy','',regex=True)
assert reference_file in mutants_esmfold.to_list(), f"reference_file must be one of {fn_esmfold}"
# Get the number of proteins
num_mutants = len(mutants_esmfold)
print(f"Number of proteins: {num_mutants}")

# (ii) Load the polypeptides
path_polypeptides = os.path.join(dir_data, 'cftr_polypeptides.csv')
df_polypeptides = pd.read_csv(path_polypeptides)
# Determine the duplicate polypeptides
df_polypeptides['residue'] = df_polypeptides['residue'].str.split('_',n=1,expand=True,regex=False)[0]
df_polypeptides['length'] = df_polypeptides['residue'].str.len()
# Create a dictionary that goes from residue to mutation, and vice-versa
residue2mutation = df_polypeptides.groupby('residue')['mutation'].apply(list).to_dict()
mutation2residue = df_polypeptides.set_index('mutation')['residue'].to_dict()
# Look for duplicate residues
dat_polydups = pd.concat([pd.DataFrame(v).assign(group=i) for i,v in enumerate(residue2mutation.values()) if len(v) > 1])
dat_polydups.rename(columns={0:'mutation'}, inplace=True)
# Determine the "reference" residue
dat_polydups['is_ref'] = dat_polydups['mutation'].isin(mutants_esmfold)
group_counts = dat_polydups.groupby('group')['is_ref'].sum().reset_index()
assert group_counts['is_ref'].max() <= 1, "There are multiple reference residues"
dat_polydups = dat_polydups.merge(df_polypeptides[['mutation','length']])
# Remove groups with no reference
dat_polydups = dat_polydups[dat_polydups['group'].isin(group_counts[group_counts['is_ref'] == 1]['group'])]
# Remove groups that are part of the "base" (i.e. they have synomymous mutations)
dat_polydups = dat_polydups[dat_polydups['group'] != dat_polydups.loc[dat_polydups['mutation'] == reference_file,'group'].values[0]]
dat_polydups.reset_index(drop=True, inplace=True)

# (iii) Load the clinical outcomes data
path_uni = os.path.join(dir_data, 'cftr2_uni.csv')
df_uni = pd.read_csv(path_uni)
path_comb = os.path.join(dir_data, 'cftr2_comb.csv')
df_comb = pd.read_csv(path_comb)
# Extract the average category and ensure they align
average_uni = df_uni.loc[df_uni['msr'] == 'average',['mutation','category','age','n_pat','value']]
assert np.all(average_uni.groupby(['category','age']).agg({'n_pat':'nunique','value':'nunique'}) == 1), "The average categories do not align"
average_uni = average_uni.groupby(['category','age','n_pat','value']).size().reset_index().drop(columns=[0])
average_uni['n_pat'] = average_uni['n_pat'].astype(int)
# Repeat for the allele combinations
average_comb = df_comb[df_comb['msr'] == 'average'].groupby(['category','age','n_pat','value']).size().reset_index().drop(columns=[0])
assert np.all(average_uni == average_comb), "The average categories do not align"
# Clean up the categories
average_uni = process_lung_range(average_uni)
    

##############################
# --- (2) PROCESS LABELS --- #

# (i) Get the y-labels for each
y_uni = process_cftr2_mutant(df_uni, ['mutation'])
y_comb = process_cftr2_mutant(df_comb, ['mutation1', 'mutation2'])
# Combine for summary statistics
y_both = pd.concat([y_uni.assign(is_pair=False), y_comb.assign(mutation=lambda x: x['mutation1']+'~'+x['mutation2']).drop(columns=['mutation1','mutation2'])],axis=0)
y_both['is_pair'].fillna(True, inplace=True)
y_both.reset_index(drop=True, inplace=True)
# Double check that if the value is 0, then the number of patients is 0 (and then set to null)
assert (y_both.loc[y_both['value']==0,'n_pat'] == 0).all(), "There are values that are 0 but have non-zero number of patients"
y_both.loc[y_both['n_pat']==0,'value'] = np.nan
# Add an aggregate category for "lung"
y_both = y_both.assign(category_agg=lambda x: np.where(x['category'].str.contains('lung'),'lung',x['category']))

# (ii) For each category, look at the number of patients, and whether they have a measurement
assert (y_both.groupby(['is_pair','mutation','category','age']).size() == 1).all(), "There are duplicate categories"
assert y_both.groupby(['is_pair','mutation']).size().shape[0] == y_both.groupby(['mutation']).size().shape[0], 'is_pair should be redundant'
missing_rate_cat_any = y_both.groupby(['is_pair','mutation','category_agg'])['value'].apply(lambda x: x.notnull().any()).astype(int)
missing_rate_any = missing_rate_cat_any.groupby(['is_pair','mutation']).sum().reset_index()
missing_rate_any_agg = missing_rate_any.groupby(['is_pair','value']).size().reset_index().rename(columns={0:'n'}).rename(columns={'value':'num_present'})
missing_rate_any_agg = missing_rate_any_agg.merge(missing_rate_any_agg.groupby('is_pair')['n'].sum().reset_index().rename(columns={'n':'tot'})).assign(pct=lambda x: x['n']/x['tot']).drop(columns='tot')
missing_rate_cat = missing_rate_cat_any.reset_index().groupby(['is_pair','category_agg','value']).size().reset_index().rename(columns={0:'n'})
missing_rate_cat_pct = missing_rate_cat.pivot(['is_pair','category_agg'],'value','n').assign(pct=lambda x: x[1]/x.sum(axis=1)).drop(columns=[0,1]).reset_index()
missing_rate_cat_pct = missing_rate_cat_pct.merge(missing_rate_cat.query('value==1'),'left').drop(columns='value')
missing_rate_cat['value'] = missing_rate_cat['value'].map({0:'Missing',1:'Present'})

# (iii) How many bi-ellelic mutations have measurement data?
y_biallele = y_comb.assign(value=lambda x: x['value'].notnull()).groupby(['mutation1','mutation2'])['value'].any()
y_biallele = y_biallele[y_biallele].reset_index().drop(columns='value')
y_biallele = pd.concat([y_biallele, y_biallele.rename(columns={'mutation1':'mutation2','mutation2':'mutation1'})]).drop_duplicates().sort_values('mutation1').reset_index(drop=True)
y_biallele_n = y_biallele.assign(homo=lambda x: x['mutation1'] == x['mutation2']).groupby(['mutation1','homo']).size().reset_index().pivot('mutation1','homo',0).fillna(0).astype(int)
ord_mutation = y_biallele_n.sum(1).sort_values().index
y_biallele_n = y_biallele_n.reset_index().rename(columns={'mutation1':'mutation'}).assign(mutation=lambda x: pd.Categorical(x['mutation'], ord_mutation))
y_biallele_n['xidx'] = y_biallele_n['mutation'].cat.codes
y_biallele_n = y_biallele_n.melt(['mutation','xidx'],value_name='n').query('n>0')

# (iv) Combine available labels for uni- and bi-allelic mutations
y_label = y_both.query('value.notnull()').drop(columns='n_pat')
y_label = y_label.assign(category=lambda x: np.where(x['category_agg']=='lung',x['category']+'_'+x['age'].str.replace('[\\<\\>\\~]','',regex=True),x['category'])).drop(columns=['age','category_agg'])
# Since Pair1-Pair2, is arbitrary, include Pair2-Pair1
tmp1, tmp2 = y_label.query('is_pair').copy(), y_label.query('~is_pair').copy()
tmp3 = tmp1['mutation'].str.split('\\~',n=1,expand=True).rename(columns={0:'mutation1', 1:'mutation2'})
tmp4 = pd.concat(objs=[tmp1.drop(columns='mutation'),tmp3],axis=1)
tmp5 = pd.concat(objs=[tmp4,tmp4.rename(columns={'mutation1':'mutation2','mutation2':'mutation1'})],axis=0).drop_duplicates()
tmp5 = tmp5.assign(is_homo=lambda x: x['mutation1']==x['mutation2'])
tmp5.rename(columns={'mutation1':'mutation'}, inplace=True)
y_label = pd.concat(objs=[tmp2, tmp5],axis=0)
# If mutation2 is missing, set to NA
y_label['mutation2'].fillna('NA', inplace=True)
# Assign whether we are homozygous, hetero, or NA (for integratated e.g. average)
y_label['is_homo'] = y_label['is_homo'].map({True:'homo',False:'hetero',np.nan:'NA'})
y_label['is_pair'] = y_label['is_pair'].map({False:'Integrated',True:'Paired'})
# Get the label count
y_label = y_label.assign(label_num=lambda x: x.groupby(['mutation','category','is_pair','is_homo']).cumcount()+1)
y_label = y_label.pivot(['mutation','mutation2','label_num'],['category','is_pair','is_homo'],'value')
# Save for later
y_label.to_csv(os.path.join(dir_data, 'y_label.csv'),index=True)


########################
# --- (3) PLOTTING --- #

# (i) Plot the number of labels we have
gg_missing_rate_cat = (pn.ggplot(missing_rate_cat, pn.aes(x='category_agg', y='n', fill='value')) +
                        pn.theme_bw() + pn.geom_col(position='dodge',color='black') + 
                        pn.scale_fill_discrete(name='Measurement') + 
                        pn.labs(y='Number of mutations', x='Measurement') + 
                        pn.theme(axis_text_x=pn.element_text(rotation=90)) + 
                        pn.ggtitle('Label missingness by category') + 
                        pn.facet_wrap('~is_pair', ncol=1, scales='free_y',labeller=pn.label_both) +
                        pn.geom_text(pn.aes(y='n*1.05',x='category_agg',label='100*pct'),format_string='{:.0f}%', size=8, data=missing_rate_cat_pct,nudge_x=0.2,inherit_aes=False))
gg_missing_rate_cat.save(os.path.join(dir_figures, 'y_num_labels_by_cat.png'), dpi=300, height=8, width=6)

# (ii) Count the number of mutants by the number of categories that have a measurement
posd = pn.position_dodge(1)
gg_missing_rate_any = (pn.ggplot(missing_rate_any_agg, pn.aes(x='num_present', y='pct', fill='is_pair')) + 
                        pn.theme_bw() + 
                        pn.geom_bar(stat='identity', position=posd, color='black') +
                        pn.scale_fill_discrete(name='Paired alleles?') + 
                        pn.scale_color_discrete(name='Paired alleles?') + 
                        pn.ggtitle('Coloured text shows number of genotypes') + 
                        pn.scale_y_continuous(labels=percent_format()) +
                        pn.labs(y='Percent of mutations', x='Number of measurements') +
                        pn.geom_text(pn.aes(y='pct+0.03',label='n',color='is_pair'), size=8, position=posd))
gg_missing_rate_any.save(os.path.join(dir_figures, 'y_num_labels_by_mutation.png'), dpi=300, height=4, width=6)

# (iii) Plot the number of pairs that different mutations have
gg_paired_allele_n = (pn.ggplot(y_biallele_n, pn.aes(x='xidx',y='n',fill='homo')) + 
    pn.theme_bw() + pn.scale_y_log10() + pn.geom_col(position='stack') + 
    pn.labs(x='Mutation (ordered)',y='# of pairs'))
gg_paired_allele_n.save(os.path.join(dir_figures, 'y_num_paired_alleles.png'), dpi=300, height=4, width=6)


##############################################
# --- (4) CALCULATE X (EMBEDDING) MATRIX --- #

stime = time()
holder = []
for i, mutant in enumerate(mutants_esmfold):
    # Load the embeddings
    path_mutant = os.path.join(dir_esmfold, f"{mutant}.npy")
    embeddings_mutant = np.load(path_mutant, allow_pickle=True).tolist()
    # Get the moments
    moments_mutant = get_embedding_moments(embeddings_mutant)
    # Convert to a dataframe
    df_mutant = embedding_to_df(moments_mutant)
    df_mutant.index = [mutant]
    df_mutant.index.name = 'mutation'
    assert not np.isnan(df_mutant.values.flatten()).any(), f"df_mutant contains NaNs for {mutant}"
    # Append dataframe
    holder.append(df_mutant)

    # Print the run-time and ETA
    dtime, nleft = time()-stime, num_mutants-(i+1)
    rate = dtime/(i+1)
    eta = rate*nleft
    print(f"Mutant: {mutant} ({i+1}/{num_mutants}) | ETA: {eta/60:.1f} minutes")
# Concatenate the dataframes
df = pd.concat(holder, axis=0)
# Add on the "duplicates" 
store_dups = []
for i, r in dat_polydups.groupby('group'):
    col_ref = r[r['is_ref'] == True]['mutation'].values[0]
    col_dups = r[r['is_ref'] != True]['mutation'].to_list()
    tmp_vals = df.loc[col_ref]
    tmp_row = pd.DataFrame(np.tile(tmp_vals.values, (len(col_dups),1)),columns=tmp_vals.index,index=col_dups)
    store_dups.append(tmp_row)
extra_rows = pd.concat(store_dups, axis=0)
extra_rows.index.name = 'mutation'
# Add on the "duplicates"
df_embeddings = pd.concat([df, extra_rows])
# Save the dataframe
path_embeddings = os.path.join(dir_data, f"mutant_embeddings.csv")
df_embeddings.to_csv(path_embeddings, index=True)


####################################
# --- (5) DISTANCE TO WILDTYPE --- #

# (i) Load the reference type
path_ref = os.path.join(dir_esmfold, f"{reference_file}.npy")
embeddings_ref = np.load(path_ref, allow_pickle=True).tolist()

# (ii) Loop over other types
stime = time()
holder = []
for i, mutant in enumerate(mutants_esmfold):
    # Load the embeddings
    path_mutant = os.path.join(dir_esmfold, f"{mutant}.npy")
    embeddings_mutant = np.load(path_mutant, allow_pickle=True).tolist()

    # Get the cosine similarity
    sim_mutant = diff_btw_matrics(embeddings_ref, embeddings_mutant)
    sim_mutant.index = [mutant]
    sim_mutant.index.name = 'mutation'
    assert np.all(sim_mutant.notnull().values.flatten()),f"df_mutant contains NaNs for {mutant}"
    # Append dataframe
    holder.append(sim_mutant)

    # Print the run-time and ETA
    dtime, nleft = time()-stime, num_mutants-(i+1)
    rate = dtime/(i+1)
    eta = rate*nleft
    print(f"Mutant: {mutant} ({i+1}/{num_mutants}) | ETA: {eta/60:.1f} minutes")
# Concatenate the dataframes
df_cosine = pd.concat(holder, axis=0)
# Save the dataframe
path_cosine = os.path.join(dir_data, "mutant_cosine.csv")
df_cosine.to_csv(path_cosine, index=True)


print('~~~ End of 6_process_Xy.py ~~~')