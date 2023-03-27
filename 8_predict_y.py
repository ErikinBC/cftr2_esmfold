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


#########################
# --- (1) LOAD DATA --- #

# (i) Load the multi-indexed y-values
y_adj = pd.read_csv(os.path.join(dir_data, 'y_adjusted.csv'),index_col=[0], header=[0,1,2])
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
    algorition = mdl()
    algorition.fit(x_train, y_train)
    yhat_test = pd.DataFrame(algorition.predict(x_test), columns=y_test.columns, index=y_test.index)
    holder_idx.append(yhat_test)
# Merge and save
res_oof = pd.concat(holder_idx)
dat_scatter = merge_pairwise_dfs(y_err.loc[res_oof.index], res_oof)
dat_scatter.rename(columns={'value_x':'y', 'value_y':'yhat'}, inplace=True)
dat_scatter.to_csv(os.path.join(dir_data,'pred_res_y.csv'), index=True)


#######################
# --- (3) RESULTS --- #

# Load in modules to calculate results
import plotnine as pn
from parameters import n_boot, alpha
from utilities.processing import di_ylbl, di_category
from utilities.stats import get_perf_msrs, bootstrap_function

# (i) Clean up the names for plotting
tmp_df = dat_scatter['cn'].str.split('\\_',regex=True,n=1,expand=True).rename(columns={0:'category',1:'lbl'})
tmp_df['lbl'] = tmp_df['lbl'].map(di_ylbl)
tmp_df['category'] = tmp_df['category'].map(di_category)
dat_eval = pd.concat(objs=[tmp_df, dat_scatter.drop(columns='cn')],axis=1)

# (ii) Get the point estimate and CIs around correlation performance
cn_gg = ['category','lbl']
res_fold = bootstrap_function(dat_eval, get_perf_msrs, 'value', cn_gg, n_boot, alpha,{'cn_gg':cn_gg, 'cn_y':'y', 'cn_yhat':'yhat'})
res_fold = res_fold[res_fold['msr'] != 'r2']
res_fold['msr'] = res_fold['msr'].map({'rho':'Spearman', 'tau':'Kendall'})

# (iii) Plot the predicted vs actual values
tmp_txt = dat_eval.query('idx.isin(["F508del","R347H"])').copy()
gg_oof_scatter = (pn.ggplot(dat_eval, pn.aes(x='yhat',y='y')) + 
    pn.theme_bw() + pn.geom_point(size=0.5) + 
    pn.labs(x='Predicted (out-of-fold)',y='Actual') + 
    pn.geom_text(pn.aes(label='idx'),color='red',data=tmp_txt,size=8,adjust_text={'expand_points':(3,3)}) + 
    pn.facet_grid('category~lbl',scales='free') + 
    pn.ggtitle('Predicted (out-of-fold) vs actual adjusted phenotype values') + 
    pn.geom_smooth(method='lm',se=False))
gg_oof_scatter.save(os.path.join(dir_figures, 'oof_scatter.png'),width=9,height=5)


# (iv) Plot the overall performance
posd = pn.position_dodge(0.5)
gg_oof_ffn = (pn.ggplot(res_fold, pn.aes(x='category',y='value',color='lbl',shape='msr')) + 
    pn.theme_bw() + pn.labs(y='Value',x='Phenotype category') + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.ggtitle('Out-of-fold correlation for adjusted phenotype values') + 
    pn.scale_color_discrete(name='Label type') + 
    pn.scale_shape_discrete(name='Correlation') + 
    pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.theme( axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_y_continuous(labels=percent_format()))
gg_oof_ffn.save(os.path.join(dir_figures, 'oof_perf.png'),width=5.5,height=3.5)


print('~~~ End of 9_predict_y.py ~~~')