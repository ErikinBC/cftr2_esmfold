"""
Figures used for paper
"""

import os
import pandas as pd
import numpy as np
import plotnine as pn
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
from mizani.formatters import percent_format
from parameters import dir_data, dir_figures, n_boot, seed
from utilities.utils import cat_from_map
from utilities.stats import get_perf_msrs, bootstrap_function

idx = pd.IndexSlice
alpha = 0.05
dir_data = 'data'
di_msrs = {'rho':"Spearman's rho", 'tau':"Kendall's tau", 'r2':"R-squared"}

# --- (1) LOAD PRED VS ACT --- #
df_contrib_wide = pd.read_csv(os.path.join(dir_data, 'dat_contrib_wide.csv'))
# Sort by predicted to adjusted, and stacked to final
df_contrib_wide = df_contrib_wide.assign(y_adj=lambda x: x['y']-x['nw'])
df_contrib_wide.rename(columns={'nnet':'pred_adj', 'stacked':'pred_full', 'y':'y_full'}, inplace=True)
cn_gg = ['mutation','category','lbl']
df_contrib_long = df_contrib_wide.melt(cn_gg,['pred_adj','pred_full','y_full','y_adj'],'tmp','val')
tmp = df_contrib_long['tmp'].str.split('_')
df_contrib_long = df_contrib_long.assign(msr=tmp.str[0], type=tmp.str[1])
df_contrib_long = df_contrib_long.pivot(cn_gg+['type'],'msr','val').reset_index()

# --- (2) CALCULATE PERFORMANCE --- #
di_perf_msrs = {'cn_gg':['category','lbl','type'], 'cn_y':'y', 'cn_yhat':'pred',
                'add_pearson':False, 'add_somersd':False,
                'add_r2':True, 'lm_r2':True, 'adj_r2':True}
assert isinstance(get_perf_msrs(df_contrib_long, **di_perf_msrs),pd.DataFrame)
res_perf_msrs = bootstrap_function(df_contrib_long, get_perf_msrs, 'value', di_perf_msrs['cn_gg'], n_boot, alpha, di_perf_msrs, True)

# --- (3) PLOT ALL PERFORMANCES --- #
res_perf_msrs_cat = res_perf_msrs.assign(type=lambda x: x['type'].map({'adj':'Adjusted', 'full':'Full'}))
res_perf_msrs_cat = res_perf_msrs_cat.assign(msr=lambda x: cat_from_map(x['msr'], di_msrs))
posd = pn.position_dodge(0.5)
gtit = 'Highlighted points are statistically significant'
gg_perf_r2_all = (pn.ggplot(res_perf_msrs_cat, pn.aes(x='msr',y='value',color='lbl',shape='type',alpha='is_sig')) + 
    pn.theme_bw() + pn.ggtitle(gtit) + 
    pn.labs(y='Value',x='Performance (%)') + 
    pn.facet_wrap('~category',scales='fixed') + 
    pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.scale_color_discrete(name='Label type') + 
    pn.scale_shape_discrete(name='Y-label') + 
    pn.scale_alpha_manual(name='Significant',values=[0.3,1]) + 
    pn.guides(alpha=False) + 
    pn.theme( axis_text_x=pn.element_text(angle=90)) + 
    pn.scale_y_continuous(labels=percent_format()))
gg_perf_r2_all.save(os.path.join(dir_figures, 'gg_perf_r2_all.png'),width=10,height=3)

# --- (4) PUBLICATION FIGURES & TABLES --- #
dat_pivot = res_perf_msrs_cat.query('type=="Full"').drop(columns='type').reset_index(drop=True)
dat_pivot = dat_pivot.assign(pval=lambda x: 2*norm.cdf(-np.abs(x['value']/x['se'])))
dat_pivot = dat_pivot.assign(fdr=lambda x: fdrcorrection(x['pval'])[1])
dat_pivot[['pval','fdr']].agg({'count','median',lambda x: (x<0.05).sum()})
dat_pivot = dat_pivot.assign(fancy_val=lambda x: (x['value']*100).round(1).astype(str)+'% (' + '±' + (x['se']*100).round(1).astype(str) + ')')
dat_pivot = dat_pivot.assign(fancy_pval=lambda x: (x['fdr']).round(2).astype(str))
dat_pivot = dat_pivot.assign(fancy_pval=lambda x: x['fancy_pval'].replace('0.0','<0.01'))
dat_pivot = dat_pivot.pivot(['category','lbl'],['msr'],['fancy_val', 'fancy_pval'])
dat_pivot.index.names = ['Category', 'Genotype']
dat_pivot.columns.names = ['Measure', 'Performance']
dat_pivot = dat_pivot.rename(columns={'fancy_pval':'P-value','fancy_val':'Value'})
dat_pivot = dat_pivot.reorder_levels(['Performance','Measure'],axis=1)
dat_pivot = dat_pivot.loc[:,idx[list(di_msrs.values())]]
dat_pivot.to_html('figtbl.html')


# --- (5) CI & PVals --- #
# Generate draws of the data
dist_res = norm(loc=res_perf_msrs_cat['value'],scale=res_perf_msrs_cat['se'])
df_sim_msr = pd.DataFrame(dist_res.rvs([n_boot,res_perf_msrs_cat.shape[0]], random_state=seed).T,index=pd.MultiIndex.from_frame(res_perf_msrs_cat[['category','lbl','type','msr']]))
df_sim_msr = df_sim_msr.melt(ignore_index=False,var_name='sim')
# (i) Across all phenotypes, what is SD of the average performance?
dist_msrs = res_perf_msrs_cat.groupby('msr')['value'].mean().reset_index().merge(df_sim_msr.groupby(['msr','sim'])['value'].mean().groupby('msr').std().reset_index().rename(columns={'value':'se'})).set_index('msr')
print((dist_msrs*100).round(1))