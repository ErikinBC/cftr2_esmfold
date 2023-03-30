"""
This is the main model fitting script that uses the ML-model class as defined in the model.py script to make predictions on the amino-acid adjusted length outcome stored in the y_adjusted.csv from the 8_debias_y.py script. 

The embeddings from both the coordinate-wise averages and cosine similarities are concatenated together to provide a total of 23,333 features. The mutations are randomly split into 5-folds, and the out-of-fold predictions are saved (data/pred_res_y.csv) and evaluated at the end the script.

This script also produces two figures:
1. oof_scatter.png: Shows the out-of-fold predicted vs actual for the adjusted label
2. oof_perf.png: Shows the average correlation and CI for the different label types, categories, and and correlation types
"""

# External modules
import os
import numpy as np
import pandas as pd
from mizani.formatters import percent_format
from sklearn.model_selection import KFold
# Local utils
from utilities.model import mdl
from utilities.utils import merge_pairwise_dfs
from parameters import dir_data, dir_figures, seed, n_folds

# Clean up for performance
di_perf_msr = {'pearson':'Pearson', 'somersd':"Somer's D", 'rho':'Spearman'} # , 'tau':'Kendall', 'r2':'R-squared'


#########################
# --- (1) LOAD DATA --- #

# (i) Load the multi-indexed y-values
y_adj = pd.read_csv(os.path.join(dir_data, 'y_adjusted.csv'),index_col=[0], header=[0,1,2])
y_adj.columns.names = pd.Series(y_adj.columns.names).replace({None:'decomp'})
# Extract the "error"
y_err = y_adj.xs('err',1,0)
y_err.columns = y_err.columns.map('_'.join)
y_mutations = pd.Series(y_err.index)
cn_y = y_err.columns
di_idx2cn = dict(zip(range(len(cn_y)),cn_y))

# (ii) Load the Xdata
xpath1 = os.path.join(dir_data, f'mutant_embeddings.csv')
xpath2 = os.path.join(dir_data, f'mutant_cosine.csv')
df_x = pd.read_csv(xpath1).merge(pd.read_csv(xpath2), on='mutation')
df_x.set_index('mutation', inplace=True)
x_mutations = df_x.index.to_list()
mutations = pd.Series(np.intersect1d(y_mutations, x_mutations))

# (iii) Subset
df_x = df_x.loc[mutations]
y_err = y_err.loc[mutations]


#################################
# --- (2) K-FOLD PREDICTION --- #

# For storage
path_pred = os.path.join(dir_data,'pred_res_y.csv')
# Loop over KFold
splitter = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
holder_idx = []
for fold, idx in enumerate(splitter.split(X=mutations)):
    print(f'---- Fold = {fold+1} ----')
    # spit the data into train/test
    ridx, tidx = idx[0], idx[1]
    rmuts, tmuts = mutations[ridx], mutations[tidx]
    y_train = y_err.loc[rmuts]
    y_test = y_err.loc[tmuts]
    x_train = df_x.loc[rmuts].copy()
    x_test = df_x.loc[tmuts].copy()

    # Fit model
    algorithm = mdl()
    algorithm.fit(x_train, y_train)
    yhat_test = pd.DataFrame(algorithm.predict(x_test), columns=y_test.columns, index=y_test.index)
    holder_idx.append(yhat_test)
# Merge and save
res_oof = pd.concat(holder_idx)
dat_scatter = merge_pairwise_dfs(y_err.loc[res_oof.index], res_oof)
dat_scatter.rename(columns={'value_x':'y', 'value_y':'yhat'}, inplace=True)
dat_scatter.to_csv(path_pred, index=True)
# Print the algorithm size
if 'algorithm' in dir():
    if all([v in dir(algorithm.algorithm) for v in ['intercepts_','coefs_']]):
        n_params = sum([i.shape[0] for i in algorithm.algorithm.intercepts_])
        n_params += sum([np.prod(c.shape) for c in algorithm.algorithm.coefs_])
        print(f'Model has a total of {n_params:,} parameters')
if not 'dat_scatter' in dir():
    dat_scatter = pd.read_csv(path_pred)


#######################
# --- (3) RESULTS --- #

# Load in modules to calculate results
import plotnine as pn
from parameters import n_boot, alpha
from utilities.utils import cat_from_map
from utilities.processing import di_ylbl, di_category
from utilities.stats import get_perf_msrs, bootstrap_function, get_perf_diff

# (i) Clean up the names for plotting
tmp_df = dat_scatter['cn'].str.split('\\_',regex=True)
tmp_df = pd.DataFrame({'category':tmp_df.str[:-1].map('_'.join), 'lbl':tmp_df.str[-1]})
tmp_df['lbl'] = tmp_df['lbl'].map(di_ylbl)
tmp_df['category'] = tmp_df['category'].map(di_category)
dat_eval = pd.concat(objs=[tmp_df, dat_scatter.drop(columns='cn')],axis=1)

# (ii) Get the point estimate and CIs around correlation performance
cn_gg = ['category','lbl']
res_fold = bootstrap_function(dat_eval, get_perf_msrs, 'value', cn_gg, n_boot, alpha,{'cn_gg':cn_gg, 'cn_y':'y', 'cn_yhat':'yhat', 'add_pearson':True, 'add_somersd':'True'})
res_fold.to_csv(os.path.join(dir_data, 'res_oof.csv'), index=False)
res_fold_cat = res_fold.assign(msr=lambda x: cat_from_map(x['msr'], di_perf_msr))
res_fold_cat = res_fold_cat.dropna().reset_index(drop=True)
(100*res_fold_cat.groupby('msr')['value'].agg({'mean','median','min','max'}).round(2)).astype(int).astype(str)+'%'

# (iii) Plot the predicted vs actual values
tmp_txt = dat_eval.query('idx.isin(["F508del","R347H"])').copy()
h = 1.75*dat_eval['category'].nunique()
gg_oof_scatter = (pn.ggplot(dat_eval, pn.aes(x='yhat',y='y')) + 
    pn.theme_bw() + pn.geom_point(size=0.5) + 
    pn.labs(x='Predicted (out-of-fold)',y='Actual') + 
    pn.geom_text(pn.aes(label='idx'),color='red',data=tmp_txt,size=8,adjust_text={'expand_points':(3,3)}) + 
    pn.facet_grid('category~lbl',scales='free') + 
    pn.ggtitle('Predicted (out-of-fold) vs actual adjusted phenotype values') + 
    pn.geom_smooth(method='lm',se=False))
gg_oof_scatter.save(os.path.join(dir_figures, 'oof_scatter.png'),width=9,height=h)

# (iv) Plot the overall performance
posd = pn.position_dodge(0.5)
nudge = 0.05
txt_res = res_fold_cat.groupby('category').agg({'value':['min','max'], 'lb':'min', 'ub':'max'})
tmp1 = txt_res.xs('min',1,1).rename(columns={'lb':'y'}).set_index('value',append=True)
tmp2 = txt_res.xs('max',1,1).rename(columns={'ub':'y'}).set_index('value',append=True)
txt_res = pd.concat(objs=[tmp1-nudge, tmp2+nudge]).reset_index()
gg_oof_ffn = (pn.ggplot(res_fold_cat, pn.aes(x='category',y='value',color='lbl',shape='msr')) + 
    pn.theme_bw() + pn.labs(y='Value',x='Phenotype category') + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.ggtitle('Out-of-fold correlation for adjusted phenotype values\nText shows min-max point estimate range') + 
    pn.scale_color_discrete(name='Label type') + 
    pn.scale_shape_discrete(name='Correlation') + 
    pn.geom_point(position=posd) + 
    pn.geom_text(pn.aes(y='y',label='100*value',x='category'),size=8,format_string='{:.0f}%',inherit_aes=False,data=txt_res) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.theme( axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_y_continuous(labels=percent_format()))
gg_oof_ffn.save(os.path.join(dir_figures, 'oof_perf.png'),width=5.5,height=3.5)


#####################################
# --- (4) RELATIVE CONTRIBUTION --- #

# Merge the data and check
tmp1 = y_adj.melt(ignore_index=False).dropna().assign(cn=lambda x: x['category']+'_'+x['lbl']).drop(columns=['category','lbl'])
tmp1 = tmp1.reset_index().pivot(['mutation','cn'],'decomp','value').reset_index().rename(columns={'yhat':'nw'})
tmp2 = dat_scatter.rename(columns={'idx':'mutation','y':'err','yhat':'nnet'})
df_contrib = tmp2.merge(tmp1, 'left', on=['mutation','cn'], suffixes=('_scatter','_adj'))
assert np.abs(df_contrib['err_scatter'] - df_contrib['err_adj']).max() <= 1e-12, 'Expected values to be the same'
df_contrib = df_contrib.drop(columns='err_scatter').rename(columns={'err_adj':'err'})
# compare y ~ nw to y ~ nw+nnet
df_contrib = df_contrib.assign(stacked=lambda x: x['nw']+x['nnet'])
df_contrib = df_contrib.melt(['mutation','cn','y'],['nw','stacked'],'mdl','yhat')
# Clean up the category/label
tmp_df = df_contrib['cn'].str.split('\\_',regex=True)
tmp_df = pd.DataFrame({'category':tmp_df.str[:-1].map('_'.join), 'lbl':tmp_df.str[-1]})
tmp_df['lbl'] = tmp_df['lbl'].map(di_ylbl)
tmp_df['category'] = tmp_df['category'].map(di_category)
df_contrib = pd.concat(objs=[tmp_df, df_contrib.drop(columns='cn')],axis=1)

# Prepare name arguments
cn_gg = ['category','lbl','mdl']
di_args_perf = {'cn_gg':cn_gg,'cn_y':'y','cn_yhat':'yhat','add_pearson':True,'add_somersd':True}
cn_index = ['category','lbl','msr']
cn_col='mdl'
cn_val='value'
di_args_diff = {'function':get_perf_msrs, 'di_args':di_args_perf, 'cn_index':cn_index, 'cn_col':cn_col, 'cn_val':cn_val}
# Get the distribution over differences
res_diff = bootstrap_function(df_contrib, get_perf_diff, cn_val='value', cn_gg=['category','lbl'], n_boot=n_boot, di_args=di_args_diff, verbose=True)


res_diff.query('category=="Infection Rate" & msr=="pearson"').drop(columns=['category','msr','se','is_sig'])

# Get the baseline and on on differences
res_bl = get_perf_msrs(df_contrib, **di_args_perf).pivot(cn_index, cn_col, cn_val)
res_bl_comp = res_bl.reset_index().merge(res_diff)
res_bl_comp = res_bl_comp.assign(lb=lambda x: x['nw']+x['lb']).assign(check=lambda x: x['lb'] < x['stacked'])
assert res_bl_comp['check'].all(), 'Lower bound should not be greater than model point estimate'
res_bl_comp = res_bl_comp.assign(ub=lambda x: x['nw']+x['ub']).assign(check=lambda x: x['ub'] > x['stacked'])
assert res_bl_comp['check'].all(), 'Upper bound should not be greater than model point estimate'
res_bl_comp.drop(columns='check', inplace=True)
res_bl_comp = res_bl_comp.melt(cn_index+['value','lb','ub','is_sig'],res_bl.columns.to_list(),'mdl','perf')
res_bl_comp = res_bl_comp.assign(lb=lambda x: np.where(x['mdl']=='nw', np.nan, x['lb']))
res_bl_comp = res_bl_comp.assign(ub=lambda x: np.where(x['mdl']=='nw', np.nan, x['ub']))
res_bl_comp['mdl'] = res_bl_comp['mdl'].map({'nw':'NW', 'stacked':'NW+NNet'})
res_bl_comp['msr'] = cat_from_map(res_bl_comp['msr'], di_perf_msr)
res_bl_comp = res_bl_comp[res_bl_comp['msr'].notnull()]
# How much of an improvement is there for NW estimators > 0 correlation
res_txt = res_diff.merge(res_bl.query('nw>0').set_index('nw',append=True).index.to_frame(False))
res_txt['msr'] = cat_from_map(res_txt['msr'], di_perf_msr)
res_txt = res_txt[res_txt['msr'].notnull()]
res_txt = res_txt.groupby(['category','lbl'])['value'].agg({'mean','median'}).reset_index().assign(y=0.9)
print(res_txt.round(2))

# Plot the Nadarya-Watson estimator vs mdl
posd = pn.position_dodge(0.5)
gg_oof_diff = (pn.ggplot(res_bl_comp, pn.aes(x='lbl',y='perf',color='mdl',shape='msr',alpha='is_sig')) + 
    pn.theme_bw() + 
    pn.labs(y='Value',x='Label type') + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.facet_wrap('~category') + 
    pn.ggtitle('Out-of-fold performance w/ (NW+NNet) & w/o ESMFold model (NW)\nHighlighted points are stat. sig. compared to NW only\nText shows median improvement for NW estimator w/ >0 correlation') + 
    pn.scale_color_discrete(name='Model') + 
    pn.scale_shape_discrete(name='Correlation') + 
    pn.scale_alpha_manual(name='Significant',values=[0.3,1]) + 
    pn.guides(alpha=False) + 
    pn.geom_point(position=posd) + 
    pn.geom_text(pn.aes(y='y',label='100*median',x='lbl'),size=8,format_string='{:.0f}%',inherit_aes=False,data=res_txt) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.theme(axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_y_continuous(labels=percent_format()))
gg_oof_diff.save(os.path.join(dir_figures, 'oof_diff_perf.png'),width=9,height=3)


# Plot the scatter
h = 1.75*df_contrib['category'].nunique()
gg_smooth = pn.geom_smooth(method='lm',se=False, size=0.5)
gg_smooth.DEFAULT_AES['linetype'] = 'dashed'
gg_oof_diff_scatter = (pn.ggplot(df_contrib, pn.aes(x='yhat',y='y',color='mdl')) + 
    pn.theme_bw() + pn.geom_point(size=0.5) + 
    pn.labs(x='Predicted (out-of-fold)',y='Actual') + 
    pn.scale_color_discrete(name='Model') + 
    gg_smooth + 
    pn.facet_grid('category~lbl',scales='fixed') + 
    pn.ggtitle('Predicted vs actual phenotype for NW vs NW+NNet\nDashed line shows OLS fit') + 
    pn.geom_smooth(method='lm',se=False))
gg_oof_diff_scatter.save(os.path.join(dir_figures, 'oof_diff_scatter.png'),width=9,height=h)



print('~~~ End of 9_predict_y.py ~~~')