"""
This script generates the figures that are used to explore the data and runtime performance. It also prints off key data statistics used for write-up.

---Figures---
1. runtime.png: Actual runtime of the esmfold.model from 5_esm_fold.py as a function of sequence length.
2. exon_pos.png: Distribution of exonic mutations across the CFTR gene by mutation type.
3. rho_f508.png: Correlation between the the different phenotype labels for F508del heterozygotes. 
4. ydist_f508.png: An empirical cumulative distribution plot showing the rank-ordered mutations and phenotype outcomes (again for F508del heterozygotes).
5. int_f508_comp.png: A scatterplot where each dots shows the relationship between the average phenotype outcome for the 'Integrated' outcomes (x-axis) and the i) F508del heterozygote or ii) average of heterozygotes (y-axis).
6. homo_f508_comp.png: A plot showing the average phenotype outcome for homozygotes (x-axis) and the F508dek heterozygote (y-axis).
7. f508_hetero_comp.png: For each mutation that has average phenotype values for at least 2 heterozygous combination, the y-axis plots the difference in value that combination has relative to the F508del combination.
"""

# External modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
import waterfall_chart
from mizani.formatters import percent_format
# Local modules
from parameters import alpha, n_boot, dir_data, dir_figures, reference_file
from utilities.utils import get_cDNA_variant_types, bootstrap_rho, find_arrow_adjustments, cat_from_map, find_closest_match, find_unique_combinations
from utilities.processing import di_ylbl, di_category, di_vartype, cn_category, vals_catory, get_y_f508, get_y_int, get_y_hetero_ave, get_y_homo

# For multiindex slicing
idx = pd.IndexSlice


#########################
# --- (1) LOAD DATA --- #

# (i) Get the runetime data
path_runtime = os.path.join(dir_data, 'runtimes.csv')
df_runtime = pd.read_csv(path_runtime)

# (ii) Get the genomic location data (GO BACK AND ADD THE EXON NUMBER)
path_genomic = os.path.join(dir_data, 'ncbi_genome_loc.csv')
df_genomic = pd.read_csv(path_genomic)
path_exonic = os.path.join(dir_data, 'cftr_exon_locs.csv')
df_exonic = pd.read_csv(path_exonic)
df_exonic = df_exonic.groupby('exon')['idx'].agg({'min','max'})[['min','max']].rename(columns={'min':'start','max':'stop'}).reset_index()

# (iii) Get the CFTR2 variant classificaiton data
path_cftr2 = os.path.join(dir_data, 'cftr2_variants.csv')
df_cftr2 = pd.read_csv(path_cftr2, usecols=['mutation','cDNA_name','allele_freq'])
df_cftr2.rename(columns={'cDNA_name':'cDNA'}, inplace=True)

# (iv) Load the y data
path_ydata = os.path.join(dir_data, 'y_label.csv')
df_ydata = pd.read_csv(path_ydata, header=[0,1,2], index_col=[0,1,2])

# (v) Load the univariate and bivariate data
df_uni = pd.read_csv(os.path.join(dir_data, 'cftr2_uni.csv'))
df_comb = pd.read_csv(os.path.join(dir_data, 'cftr2_comb.csv'))

# (vi) Load in the CFTR1 database
di_cftr1 = {'cDNA Name':'cDNA_name', 'Protein Name':'protein_name', 'Legacy Name':'mutation', 'Region':'region'}
df_cftr1 = pd.read_csv(os.path.join(dir_data, 'cftr1.csv'))
df_cftr1 = df_cftr1[list(di_cftr1)].rename(columns=di_cftr1)
df_cftr1 = df_cftr1.drop_duplicates().reset_index(drop=True)

# (vii) Load in the up-to-date xlsx file
df_muthist = pd.read_csv(os.path.join(dir_data, 'mutation_history.csv'))
df_muthist = df_muthist.assign(in_uni=lambda x: x['legacy_name'].isin(df_uni['mutation'].unique()))

# (viii) Load the Xy data
y_label = pd.read_csv(os.path.join(dir_data, 'y_label.csv'),index_col=[0,1,2],header=[0,1,2])
xmuts1 = pd.read_csv(os.path.join(dir_data, "mutant_cosine.csv"),usecols=['mutation'])
xmuts2 = pd.read_csv(os.path.join(dir_data, "mutant_embeddings.csv"),usecols=['mutation'])
u_xmuts = pd.Series(pd.concat(objs=[xmuts1, xmuts2]).drop_duplicates()['mutation'].unique())
u_xmuts = u_xmuts[u_xmuts != reference_file].reset_index(drop=True)

# (ix) Load in the amino acid seqs used for ESMFold
df_aminos = pd.read_csv(os.path.join(dir_data, 'dat_aminos.csv'),usecols=['mutation','has_len','is_syn','is_dup'])

# (x) Load predicted results
dat_scatter = pd.read_csv(os.path.join(dir_data,'pred_res_y.csv')).drop(columns='Unnamed: 0',errors='ignore')
u_muts_pred = pd.Series(dat_scatter['idx'].unique())

# (xi) Load NCBI matched pages
dat_ncbi_loc = pd.read_csv(os.path.join(dir_data, 'ncbi_genome_loc.csv'), usecols=['mutation','from','to','links'])

# (xii) Get the exon location
cftr_gene = pd.read_csv(os.path.join(dir_data, 'cftr_exon_locs.csv'))
dat_ncbi_loc['is_intron'] = ~dat_ncbi_loc['from'].isin(cftr_gene['idx'])
print(f"There are {dat_ncbi_loc['is_intron'].sum()} intronic mutations")


#####################################
# --- (2) PRINT DATASET NUMBERS --- #

# --- (i) Number of CFTR2 mutations --- #
# Total count
u_uni = df_uni['mutation'].unique()
n_uni = len(u_uni)
print(f"There a total of {n_uni} mutations that were scraped from CFTR2")
# Which actually have labels
print('What percent had some clinical data:')
uni_any_val = df_uni.groupby(['msr','mutation']).apply(lambda x: x['value'].replace('insufficient data',np.nan).notnull().any())
uni_has_val = uni_any_val.groupby('msr').agg({'sum','count'})
print(uni_has_val)
print('\n')
assert len(np.setdiff1d(df_uni['mutation'].unique(),df_muthist['legacy_name'])) == 0, 'Scraped variants should be found in mutation history xlsx file'
# Compare the number scraped to the total available in the mutation_history xlsx file
print('Breakdown of variants there were scraped compared to mutation history file')
print(df_muthist['in_uni'].value_counts())
print(df_muthist.groupby(['in_uni','cf_causing'])['num_alleles'].agg({'count','mean','median','max'}).round(0).astype(int))
print('\n')
missing_cf_causing_muts = df_muthist.query('~in_uni & cf_causing!="Non CF-causing"')['legacy_name']
print(f'A total of {len(missing_cf_causing_muts)} mutations were missed from the scrape:')
# print("\n".join(missing_cf_causing_muts))
# print('\n')

# Get the number of combinations
n_comb = df_comb.groupby(['mutation1','mutation2']).size().reset_index().drop(columns=0)
n_comb = find_unique_combinations(n_comb, 'mutation1', 'mutation2', keep_ukey=True)
n_comb_homo = n_comb.query('mutation1 == mutation2')
n_comb_f508 = n_comb[n_comb['ukey'].str.contains("F508del")]
print(f"There a total of {n_comb.shape[0]} mutation pairs that were scraped from CFTR2, {n_comb_f508.shape[0]} are F508 combinations and {n_comb_homo.shape[0]} and homozygous\n")

# Compare the number of labels that have a heterozygotic combination, and of those, which are only F508del
dat_hetero = df_ydata.loc[:,idx[:,:,'hetero']]
dat_hetero = dat_hetero[dat_hetero.notnull().any(1)]
dat_hetero = dat_hetero.index.to_frame(False).set_index(['mutation','label_num']) == 'F508del'
dat_hetero = dat_hetero.reset_index().pivot_table('label_num','mutation','mutation2','count').fillna(0).astype(int)
dat_hetero = dat_hetero.clip(0,1).astype(str)
dat_hetero = (dat_hetero[False] + dat_hetero[True]).map({'01':'F508-only','11':'Both',00:'None','10':'Other-only'})
print(dat_hetero.value_counts())


# --- (ii) NCBI matching --- #
assert dat_ncbi_loc['mutation'].isin(u_uni).all(), 'NCBI should be a subset of what was scraped'
n_ncbi = dat_ncbi_loc['mutation'].nunique()
n_ncbi_exon = dat_ncbi_loc.query('is_intron==False').shape[0]

# --- (iii) Small cell censoring --- #
# Get the number of ylabels
u_ylbl = y_label.any(1).groupby('mutation').any()
n_ylbl = u_ylbl.sum()
assert uni_has_val.loc['mutant','sum'] == n_ylbl, 'should align with ylabel'
# See how many NCBI mutations have y labels
n_ylbl_adj = u_ylbl.index.isin(dat_ncbi_loc['mutation']).sum()

# Determine what the small-cell censoring is
n_data_muts = df_uni.query('msr=="mutant"').groupby('mutation')[['num_alleles','n_pat']].agg('max').astype(int).reset_index()
n_data_muts = n_data_muts.merge(uni_any_val.xs('mutant').reset_index().rename(columns={0:'has_val'}))
n_data_muts = n_data_muts.assign(has_val=lambda x: x['has_val'].astype(str)).pivot_table(['num_alleles','n_pat'],None,'has_val',['min','max','count'])
n_data_muts = n_data_muts.reorder_levels(['has_val',None],1).loc[:,idx[['False','True']]]
print('There appears to be a small-cell censoring of <=5')
print(n_data_muts)

# Repeat for combinations
n_data_comb = df_comb.query('msr=="mutant"').replace('insufficient data',np.nan).groupby(['mutation1','mutation2'])['value'].apply(lambda x: x.notnull().any()).reset_index()
n_data_comb_f508del = (n_data_comb.set_index('value') == 'F508del').any(1).groupby('value').sum()
print(f'Out of the {n_data_comb.shape[0]} paired mutants, {n_data_comb["value"].sum()} have at least one label, on which {n_data_comb_f508del.loc[True]} have F508del (out of {n_data_comb_f508del.sum()})')

# --- (iv) X-mutations (i.e. embeddings) --- #
assert u_xmuts.isin(df_uni['mutation'].unique()).all(), 'If there is an emedding it should have come from a previous file'
dat_xmut_in_ylbl = u_ylbl[u_ylbl.index.isin(dat_ncbi_loc['mutation'])].reset_index().drop(columns=0).assign(has_ylbl=True)
dat_xmut_in_ylbl = dat_xmut_in_ylbl.assign(has_xlbl=lambda x: x['mutation'].isin(u_xmuts))
dat_xmut_in_ylbl = dat_xmut_in_ylbl.merge(df_aminos, 'left')
assert u_muts_pred.isin(dat_xmut_in_ylbl['mutation']).all(), 'OOF predicted mutations should be a subset'
dat_xmut_in_ylbl['has_pred'] = dat_xmut_in_ylbl['mutation'].isin(u_muts_pred)
dat_xmut_in_ylbl.set_index('mutation', inplace=True)
assert (dat_xmut_in_ylbl.query('has_pred').nunique() == 1).all(), 'If there is a prediction, then the high level conditions should hold'
dat_xmut_in_ylbl.groupby(dat_xmut_in_ylbl.columns.to_list()).size()
n_has_len = dat_xmut_in_ylbl.query('has_len==True').shape[0]
syn_muts = dat_xmut_in_ylbl.query('has_len==True & is_syn==True').index.to_list()
n_syn_muts = dat_xmut_in_ylbl.query('has_len==True & is_syn==False').shape[0]
dup_muts = dat_xmut_in_ylbl.query('has_len==True & is_syn==False & is_dup==True').index.to_list()
# The reason that the duplicated mutations don't have a predicted value, is that the cosine measure didn't include a duplicate append
assert len(np.setdiff1d(xmuts1, xmuts2)) == 0, 'The cosine measurements should not have any different values'
assert len(np.setdiff1d(dup_muts, np.setdiff1d(xmuts2, xmuts1))) == 0, 'Any duplicates should be b/c the cosine measure didnt append the duplicate mutations'
n_dup_muts = dat_xmut_in_ylbl.query('has_len==True & is_syn==False & is_dup==False').shape[0]
print(f'The following {len(syn_muts)} variants should be synonymous')
tmp_print = '\n'.join([f'{k}:{v}' for k,v in dat_ncbi_loc.set_index('mutation').loc[syn_muts,'links'].to_dict().items()])
print(tmp_print)


# --- (v) Summarize waterfall chart --- #
di_waterfall = {'CFTR2':df_muthist.shape[0],
                'Scraped':n_uni,
                'NCBI':n_ncbi,
                'Exonic':n_ncbi_exon,
                'Small cell':n_ylbl_adj,
                'Amino Length':n_has_len,
                'Is Missense':n_syn_muts,
                'Duplicated Seq':n_dup_muts}
df_waterfall = pd.DataFrame.from_dict(di_waterfall, orient='index').rename_axis('data').rename(columns={0:'n_amino'})
df_waterfall = df_waterfall.assign(d_amino = lambda x: (x['n_amino']-x['n_amino'].shift(1)).fillna(x['n_amino'].max())).astype(int).reset_index()
# Make a plot showing the waterfall change in values
plt_waterfall = waterfall_chart.plot(df_waterfall['data'], df_waterfall['d_amino'], y_lab='# of genotypes', formatting='{:.0f}', rotation_value=90)
plt_waterfall.ylim(bottom=0)
plt_waterfall.savefig(os.path.join(dir_figures, 'waterfall_nsamp.png'))
plt_waterfall.close()


# --- (vi) Differences of mutation history to CFTR1 --- #
u_uni_muts = pd.Series(df_uni['mutation'].dropna().unique())
u_uni_cDNA = pd.Series(df_uni['cDNA_name'].dropna().unique())
u_cftr1_muts = pd.Series(df_cftr1['mutation'].dropna().unique())
u_cftr1_cDNA = pd.Series(df_cftr1['cDNA_name'].dropna().unique())
# Check that comb is a subset of uni
assert pd.Series(df_comb['mutation1'].unique()).isin(u_uni_muts).all(), 'combination should be a subset of uni'
assert pd.Series(df_comb['mutation2'].unique()).isin(u_uni_muts).all(), 'combination should be a subset of uni'
# Do a fuzzy match
tmp_match_muts = u_uni_muts.apply(lambda x: find_closest_match(x, u_cftr1_muts))
tmp_match_muts.index = u_uni_muts
tmp_match_cDNA = u_uni_cDNA.apply(lambda x: find_closest_match(x, u_cftr1_cDNA))
tmp_match_cDNA.index = u_uni_cDNA
tmp_match = pd.concat(objs=[tmp_match_muts.reset_index().assign(tt='mutation'), tmp_match_cDNA.reset_index().assign(tt='cDNA_name')],axis=0).rename(columns={0:'tmp'})
tmp_match = tmp_match.assign(match=lambda x: x['tmp'].str[0], cutoff=lambda x: x['tmp'].str[1]).drop(columns='tmp')
tmp_match = df_uni.groupby(['mutation','cDNA_name']).size().reset_index().drop(columns=0).rename_axis('idx').melt(value_name='index',var_name='tt',ignore_index=False).reset_index().merge(tmp_match, 'left')
# Match below 80 are all noise (.assign(match=lambda x: np.where(x['cutoff'] < 0.8, np.nan, x['match'])))
matchable_mutations = tmp_match.rename(columns={'index':'ref'}).pivot('idx','tt',['ref','match','cutoff'])
# Unless there is one 100% match, we will treat it is as missing
matchable_mutations.columns.names = ['msr', 'tt']
matchable_mutations = matchable_mutations.loc[matchable_mutations.loc[:,idx['cutoff']].max(1) == 1]
# If the cDNA is 100% use it, otherwise rely on mutation
idx_cDNA = matchable_mutations.loc[:,idx['cutoff','cDNA_name']] == 1
cDNA_matchable = matchable_mutations.loc[idx_cDNA,idx['match','cDNA_name']]
muts_matchable = matchable_mutations.loc[~idx_cDNA,idx['match','mutation']]
n_cDNA_matchable, n_muts_matchable = len(cDNA_matchable), len(muts_matchable)
n_matchable = n_cDNA_matchable + n_muts_matchable
print(f'A total of {n_matchable} mutations from CFTR2 can be linked to CFTR1 ({n_cDNA_matchable} by cDNA name, and {n_muts_matchable} by legacy name), meaning {len(u_uni_cDNA)-n_matchable} cannot be matched')
# Determine which CFTR1 have CFTR2 matches
tmp1 = df_cftr1[df_cftr1['mutation'].isin(muts_matchable)]
assert tmp1['mutation'].nunique() == n_muts_matchable, 'Expected a 1:1 match'
tmp2 = df_cftr1[df_cftr1['cDNA_name'].isin(cDNA_matchable)]
assert tmp2['cDNA_name'].nunique() == n_cDNA_matchable, 'Expected a 1:1 match'
tmp3 = pd.concat(objs=[tmp1, tmp2]).assign(cftr2=True)
tmp4 = df_cftr1[~df_cftr1.index.isin(tmp3.index)].assign(cftr2=False)
df_cftr12 = pd.concat(objs=[tmp3, tmp4], axis=0).reset_index(drop=True)
# Cleap up the region/exon
df_cftr12['region'] = df_cftr12['region'].str.split('\\s').str[0].fillna('missing')
print(df_cftr12.groupby(['cftr2','region']).size())


############################
# --- (3) PROCESS DATA --- #

# (i) Determine "order" which fits the data the best
d_orders = list(range(1,6))
df_runtime = df_runtime.assign(group=lambda x: pd.cut(x['length'],[0,500,750,1000,1250,1500]))
holder = []
for g in df_runtime['group'].unique():
    idx_train = df_runtime['group'] != g
    train, test = df_runtime[idx_train], df_runtime[~idx_train]
    perf_g = pd.concat([pd.DataFrame({'y':test['runtime'],'pred':np.poly1d(np.polyfit(x=train['length'], y=train['runtime'], deg=d))(test['length']), 'order':d}) for d in d_orders]).groupby('order').apply(lambda x: np.sqrt(np.mean((x['y'] - x['pred'])**2))).reset_index().rename(columns={0:'rmse'})
    holder.append(perf_g)
perf_orders = pd.concat(holder).reset_index(drop=True)
order_star = perf_orders.groupby('order')['rmse'].mean().idxmin()
formula_star = np.polyfit(x=df_runtime['length'], y=df_runtime['runtime'], deg=order_star)
extrap_star = np.poly1d(formula_star)

# (ii) Get exonic location of CFTR2 variants
tmp = df_exonic.assign(intron_start=lambda x: x['stop'].shift(1)).dropna().astype(int).assign(exon=lambda x: x['exon']-0.5).drop(columns='stop').rename(columns={'intron_start':'start', 'start':'stop'})
exon_intron = pd.concat([df_exonic, tmp],axis=0).sort_values('exon')
exon_intron = exon_intron.assign(stop=lambda x: np.where(x['exon'] == x['exon'].max(), x['stop']+1, x['stop']))
exon_intron['type'] = np.where(exon_intron['exon'] % 1 == 0, 'exon', 'intron')
u_vals = np.sort(np.unique(list(exon_intron['start']) + list(exon_intron['stop'])))
exon_intron['group'] = pd.cut(exon_intron['start'], u_vals, right=False)
mutation_locs = df_genomic[['mutation','from','to']].assign(group=lambda x: pd.cut(x['from'], u_vals, right=False))
mutation_locs = mutation_locs.merge(exon_intron, 'left', 'group')
mutation_locs = mutation_locs[mutation_locs['start'].notnull()]
mutation_locs = mutation_locs[mutation_locs['type'] == 'exon']
mutation_locs = mutation_locs.set_index('mutation')[['exon','from','to']].astype(int)
# Determine the relative position of the exons
tmp = df_exonic.set_index('exon').diff(axis=1)['stop'].cumsum().reset_index().assign(start=0).assign(start=lambda x: x['stop'].shift(1)).fillna(0).astype(int)[['exon','start','stop']]
rel_exonic = df_exonic.drop(columns='stop').rename(columns={'start':'rel_pos'}).merge(tmp)
# Get the relative position
mutation_locs = mutation_locs.reset_index().merge(rel_exonic, 'left', 'exon')
mutation_locs = mutation_locs.assign(x_pos=lambda x: x['from'] - x['rel_pos'] +x['start'])[['mutation','exon','from','x_pos']].sort_values(['exon','from'])
# Get allele frequency
mutation_locs = mutation_locs.merge(df_cftr2, 'left', 'mutation')
# Add on variant types
mutation_locs['vartype'] = get_cDNA_variant_types(mutation_locs['cDNA'])


#########################
# --- (4) F508 DIST --- #

# (i) Look at pairing with F508del
data_f508 = get_y_f508(df_ydata)

# (ii) What is the correlation between categories?
data_f508_wide = data_f508.pivot('mutation','category','value')
rho_f508 = pd.concat(objs=[bootstrap_rho(data_f508_wide, nboot=n_boot, alpha=alpha, method=m).assign(method=m) for m in ['pearson','kendall','spearman']],axis=0).reset_index(drop=True)
rho_f508[['category1','category2']] = rho_f508[['category1','category2']].replace(di_category)
rho_f508 = rho_f508.assign(xval=lambda x: x['category2'] + '-' + x['category1'])
rho_f508.drop(columns=['category1','category2'],inplace=True)
rho_f508['xval'] = pd.Categorical(rho_f508['xval'], rho_f508.groupby('xval')['rho'].mean().sort_values().index)

# (iii) What the distribution look like?
data_f508_dist = data_f508.sort_values(['category','value']).assign(ridx=lambda x: x.groupby('category').cumcount()+1).merge(df_cftr2, 'left')
data_f508_dist = data_f508_dist.assign(vartype=lambda x: get_cDNA_variant_types(x['cDNA']),allele_freq=lambda x: -np.log10(x['allele_freq']))
data_f508_dist.dropna(inplace=True)
data_f508_dist['category'] = cat_from_map(data_f508_dist['category'], di_category)


#####################################
# --- (5) VARIATION AROUND F508 --- #

# (i) x-axis: F508del hetero, y-axis: integrated average (1:1)
dat_int_dist = get_y_int(df_ydata)
dat_int_dist['category'] = cat_from_map(dat_int_dist['category'], di_category)
dat_int_dist_comp = dat_int_dist.merge(data_f508_dist[['mutation','category','value']],'inner',on=['mutation','category'],suffixes=('_int','_f508'))

# (ii) x-axis: average or hetero/homo, y-axis: integrated average (1:1)
dat_ave_hetero = get_y_hetero_ave(df_ydata)
dat_ave_hetero['category'] = cat_from_map(dat_ave_hetero['category'], di_category)
# Add onto the integrated
dat_int_dist_comp = dat_int_dist_comp.merge(dat_ave_hetero,'inner').rename(columns={'value':'value_pair'})
dat_int_dist_comp = dat_int_dist_comp.melt(['mutation','category','value_int'],var_name='comp').assign(comp=lambda x: x['comp'].str.replace('value_','',regex=False))

# (iii) x-axis: mutant homo, y-axis: F508del (when is homo worse the F508 hetero)
dat_homo_dist = get_y_homo(df_ydata)
dat_homo_dist['category'] = cat_from_map(dat_homo_dist['category'],di_category)
dat_homo_f508_comp = dat_homo_dist.merge(data_f508_dist[dat_homo_dist.columns],'inner',['mutation','category'],suffixes=('_homo','_f508'))

# (iv) x-axis: F508del hetero, y-axis: other hetero (1:many) - show average with alpha=1, the rest with alpha=0.25
dat_hetero_other = df_ydata.loc[:,idx[cn_category,'Paired','hetero']]
dat_hetero_other = dat_hetero_other.droplevel(level=(1,2),axis=1)
other_hetero = list(np.setdiff1d(dat_hetero_other.index.get_level_values('mutation2').fillna('').unique(),['F508del','']))
dat_hetero_other = dat_hetero_other.loc[idx[:,other_hetero,:]]
dat_hetero_other = dat_hetero_other[dat_hetero_other.index.get_level_values('mutation') != 'F508del']
dat_hetero_other = dat_hetero_other.melt(ignore_index=False).dropna().reset_index().drop(columns='label_num')
dat_hetero_other['category'] = cat_from_map(dat_hetero_other['category'],di_category)
dat_hetero_f508_comp = dat_hetero_other.merge(data_f508_dist[dat_hetero_other.columns.drop('mutation2')],'inner',on=['mutation','category'],suffixes=('_hetero','_f508'))

########################
# --- (6)  --- #




########################
# --- (7) PLOTTING --- #

# (i) Plot the GPU run-time of an A100
dat_smooth = pd.DataFrame({'length':df_runtime['length'], 'smooth':extrap_star(df_runtime['length'])})
tidy_formula = np.round(formula_star, 2)
sign_formula = np.sign(formula_star)
tidy_formula  = pd.Series(np.where(tidy_formula == 0, '0', tidy_formula.astype(str)))
tidy_formula + '*' + pd.Series(['', 'x', 'x^2', 'x^3'])

gtit_runtime = 'Runtime of CFTR2 analysis on A100 GPU\nRed line shows polynomial fit of degree 3 ~ O(n^3)'
gg_runtime = (pn.ggplot(df_runtime, pn.aes(x='length', y='runtime/60')) + 
                pn.geom_point() + pn.theme_bw() + 
                pn.ggtitle(gtit_runtime) + 
                pn.geom_line(pn.aes(x='length', y='smooth/60'), data=dat_smooth, color='red') +
                pn.labs(x='Sequence length', y='Runtime (m)'))
gg_runtime.save(os.path.join(dir_figures,'runtime.png'), width=6, height=4, dpi=300)


# (ii) Plot the position of the different variants
tmp_xaxis = rel_exonic.assign(center=lambda x: (x['stop']+x['start'])/2)
selected_mutations = mutation_locs.sort_values('allele_freq').tail(5)['mutation'].to_list() + ['R347H']
selected_mutations = mutation_locs[mutation_locs['mutation'].isin(selected_mutations)]

gg_exon_pos = (pn.ggplot(mutation_locs, pn.aes(x='x_pos', y='-np.log10(allele_freq)', color='vartype')) + 
            pn.theme_bw() + pn.geom_point() + 
            pn.scale_color_discrete(name='Variant type',labels=lambda x: [di_vartype[z] for z in x]) +
            pn.labs(x='Exon', y='-log10(allele frequency)') + 
            pn.ggtitle('CFTR2 variants') + 
            pn.geom_text(pn.aes(label='mutation'), data=selected_mutations, size=8, adjust_text={'expand_points':(2.5, 2.5),'arrowprops': {'arrowstyle': 'simple'}}) + 
            pn.theme(axis_text_x=pn.element_text(angle=90)) + 
            pn.scale_x_continuous(breaks=tmp_xaxis['center'], labels=tmp_xaxis['exon']))
gg_exon_pos.save(os.path.join(dir_figures,'exon_pos.png'), width=8, height=4, dpi=500)

# (iii) Plot the correlation between different measures
posd = pn.position_dodge(0.5)
ylb = rho_f508['lb'].min()-0.05
txt_rho_f508 = rho_f508.groupby('xval').agg({'rho':'mean','lb':'min'}).reset_index()
gg_rho_f508 = (pn.ggplot(rho_f508, pn.aes(x='xval',y='rho',color='method')) + 
    pn.theme_bw() + pn.geom_point(size=1,position=posd) + 
    pn.labs(y='Correlation coefficient') + 
    pn.scale_color_discrete(name='Method') + 
    pn.geom_text(pn.aes(y='lb',label='100*rho',x='xval'),inherit_aes=False,size=8,angle=90,format_string='{:.0f}%',data=txt_rho_f508,nudge_y=-0.1) + 
    pn.theme(axis_title_x=pn.element_blank(), axis_text=pn.element_text(angle=90)) + 
    pn.scale_y_continuous(labels=percent_format(),limits=[-1,1]) + 
    pn.geom_hline(yintercept=0,linetype='--',color='black') + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.ggtitle('Uses only F508del-heterozygotes\nText shows average correlation'))
gg_rho_f508.save(os.path.join(dir_figures,'rho_f508.png'), width=6, height=4, dpi=300)

# (iv) Look at the phenotypic distribution for patients with F508del
selected_f508_mutations = ['R347H', 'F508del']
selected_f508_txt = data_f508_dist.loc[data_f508_dist['mutation'].isin(selected_f508_mutations),['category','mutation','ridx','value']]
selected_f508_txt = find_arrow_adjustments(df=selected_f508_txt, cn_gg='category', cn_x='ridx', cn_y='value')
selected_f508_txt = selected_f508_txt.assign(ylbl=lambda x: np.maximum(x['ylbl']*1.1,x['ylbl']+10))

dat_hlines = pd.DataFrame.from_dict(di_category,orient='index').reset_index().rename(columns={'index':'msr',0:'category'})
dat_normal = pd.DataFrame.from_dict({'infection':0.01, 'PI':0.01, 'sweat':30, 'lung_max_All':120, 'lung_min_All':80, 'lung_mid_All':100},orient='index').reset_index().rename(columns={'index':'msr',0:'normal'})
dat_hlines = dat_hlines.merge(dat_normal)

gg_f508_dist = (pn.ggplot(data_f508_dist, pn.aes(x='ridx',y='value',color='allele_freq',shape='vartype')) + 
    pn.theme_bw() + pn.geom_point(size=0.25) + 
    pn.ggtitle('Average clinical outcome for patients with F508del\nHorizontal line shows "normal" population') + 
    pn.geom_text(pn.aes(label='mutation',x='xlbl',y='ylbl'),inherit_aes=False,data=selected_f508_txt, size=8) + 
    pn.geom_segment(pn.aes(x='xlbl',xend='x',y='ylbl',yend='y'),size=0.5,color='black',inherit_aes=False,data=selected_f508_txt,alpha=0.25) + 
    pn.facet_wrap('~category',scales='free') + 
    pn.geom_hline(pn.aes(yintercept='normal'),data=dat_hlines, inherit_aes=False,color='black',linetype='--') + 
    pn.theme(subplots_adjust={'hspace': 0.25, 'wspace': 0.25}) + 
    pn.scale_color_continuous(name='-log10(allele frequency)') +  
    pn.scale_shape_discrete(name='Variant type',labels=lambda x: [di_vartype[z] for z in x]) + 
    pn.labs(y='Value',x='Mutation (ordered by value)'))
gg_f508_dist.save(os.path.join(dir_figures,'ydist_f508.png'), width=8, height=5, dpi=600)


# (v) Plot the correlation between the average (or "integrated") value, and the different "paired" types
cn_gg = ['category','comp']
tmp_txt = dat_int_dist_comp.set_index('mutation').groupby(cn_gg).apply(lambda x: pd.DataFrame({'rho':x.corr().iloc[0,1], 'x':x['value_int'].quantile(0.2),'y':x['value'].quantile(0.99)},index=[0]))
tmp_txt = tmp_txt.reset_index().drop(columns='level_2').merge(tmp_txt.groupby('category')['y'].max().reset_index().rename(columns={'y':'ym'}))
tmp_txt = tmp_txt.assign(y=lambda x: np.where(x['comp']=='f508',x['ym'],x['ym']-10))

def tmp_mapping(x:list):
    return [di_ylbl[z] if z in di_ylbl else '' for z in x]

gg_int_f508_comp = (pn.ggplot(dat_int_dist_comp,pn.aes(x='value_int',y='value',color='comp')) + 
    pn.theme_bw() + pn.geom_point(size=0.5) +
    pn.scale_color_discrete(name='Paired',labels=tmp_mapping) +  
    pn.facet_wrap('~category',scales='free') + 
    pn.geom_smooth(method='lm',linetype='--',se=False) + 
    pn.labs(x='Average mutation value',y='Paired mutation value') + 
    pn.theme(subplots_adjust={'hspace': 0.25,'wspace':0.25}) +
    pn.ggtitle('Comparing the "average" clinical value to the paired values') + 
    pn.geom_text(pn.aes(label='rho*100',x='x',y='y'),size=8,data=tmp_txt,format_string='rho={:.0f}%'))
gg_int_f508_comp.save(os.path.join(dir_figures,'int_f508_comp.png'), width=9, height=6)


# (vi) Plot the difference in phenotype between the homozygous and F508 hetero, where relevant
tmp_txt = dat_homo_f508_comp.assign(err=lambda x: x['value_homo']-x['value_f508']).sort_values(['category','err']).groupby('category').apply(lambda x: pd.concat(objs=[x.head(1),x.tail(1)],axis=0)).reset_index(drop=True)
gg_homo_f508_comp = (pn.ggplot(dat_homo_f508_comp,pn.aes(x='value_homo',y='value_f508')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.ggtitle('Homozygous vs F508del-hetero\nBlue line shows x==y') +
    pn.theme(subplots_adjust={'hspace': 0.25,'wspace':0.25}) + 
    pn.geom_text(pn.aes(label='mutation'),data=tmp_txt,size=10,color='red',adjust_text={'expand_points': (3, 3)}) + 
    pn.labs(x='Homozygous',y='F508-Hetero') + 
    pn.facet_wrap('~category',scales='free') + 
    pn.geom_abline(slope=1,intercept=0,color='blue',linetype='--'))
gg_homo_f508_comp.save(os.path.join(dir_figures,'homo_f508_comp.png'), width=8, height=5)

# (vii) Compare F508 to different mutations
tmp_df = dat_hetero_f508_comp.assign(vdif=lambda x: x['value_hetero']-x['value_f508']).drop(columns=['value_f508','value_hetero'])
tmp_df['mutation'] = pd.Categorical(tmp_df['mutation'], tmp_df.groupby('mutation')['vdif'].apply(lambda x: x.abs().mean()).sort_values().index)
tmp_df = tmp_df.sort_values(['category','mutation','vdif']).reset_index(drop=True)
tmp_df['idx'] = tmp_df.mutation.cat.codes

gg_f508_hetero_comp = (pn.ggplot(tmp_df,pn.aes(x='idx',y='vdif',color='idx',group='mutation')) +  
    pn.theme_bw() + pn.geom_point(size=0.5) + 
    pn.geom_line(alpha=0.5) + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.labs(x='Mutation',y='Difference to F508del') + 
    pn.guides(color=False) + 
    pn.facet_wrap('~category',scales='free') + 
    pn.theme(subplots_adjust={'hspace': 0.25,'wspace':0.25}) + 
    pn.ggtitle('Clinical outcomes for different heterozygous combinations with F508 benchmark'))
gg_f508_hetero_comp.save(os.path.join(dir_figures,'f508_hetero_comp.png'), width=8, height=5)



print('~~~ End of 7_summary_stats.py ~~~')