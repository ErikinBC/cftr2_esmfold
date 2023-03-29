"""
This script finds a (smoothed) association between the amino acid length and the phenotypic outcome. The --ylabels flag determines which columns of the y_label.csv data will be used. The script saves as data as y_adjusted.csv where the column is a MultiIndex of three values. The first level is either 'err':y-yhat, 'yhat':predicted value, or 'y':actual label, where the yhat=f(amino_acid length). The second level ("category") is one of the label types specified by the --ylabels flag. The third level ('lbl') is one of the four outcome approaches: 'f508':F508del-heterozygous, 'int':Single-variant average, 'pair':Average heterozygous value, and 'homo':Homozygous value.

This script also save two figures to the figures/folder:
1. within_y_rho.png: Show's the within-category correlation for the labels defined in the ylabels argument (e.g. correlation between homoyzgous outcomes and F508del-hetero outcomes for sweat chloride concentration).
2. nw_ylbl.png: The actual vs LOO-NW estimator prediction and 100*(1-alpha)%-CI 
"""

# External modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from mizani.formatters import percent_format
# Internal modules
from parameters import alpha, n_boot, dir_data, dir_figures, ylabels
from utilities.utils import get_tril_corr, cat_from_map
from utilities.stats import NadarayaWatson, bootstrap_function, get_perf_msrs
from utilities.processing import get_y_f508, get_y_int, get_y_hetero_ave, get_y_homo, di_category, di_ylbl


################################
# --- (1) PREPARE Y-VALUES --- #

# (i) Load the polypeptide length
df_aminos = pd.read_csv(os.path.join(dir_data, 'cftr_polypeptides.csv'))
df_aminos['residue'] = df_aminos['residue'].apply(lambda x: x.split('_')[0],1)
df_aminos = df_aminos.assign(length=lambda x: x['residue'].apply(len)).drop(columns='residue')

# (ii) Load the y data
path_ydata = os.path.join(dir_data, 'y_label.csv')
ydata = pd.read_csv(path_ydata, header=[0,1,2], index_col=[0,1,2])

# (iii) Process the different Y-labels
y_f508 = get_y_f508(ydata).drop(columns='is_homo').assign(lbl='f508')
y_int = get_y_int(ydata).assign(lbl='int')
y_hetero = get_y_hetero_ave(ydata).assign(lbl='pair')
y_homo = get_y_homo(ydata).assign(lbl='homo')
# Remove the "lung_max_All" category
df_y = pd.concat(objs=[y_f508, y_int, y_hetero, y_homo],axis=0)
df_y = df_y[df_y['category'].isin(ylabels)]
df_y = df_y.pivot('mutation',['category','lbl'],'value')
assert df_y.notnull().any(1).all()

# (iv) Plot the different correlations
rho_y = pd.concat([bootstrap_function(df_y.xs(y,1,0).assign(grp=1).set_index('grp'), get_tril_corr, 'rho', 'grp', n_boot, alpha, {'method':['spearman','pearson','kendall']}).assign(y=y) for y in ylabels]).reset_index(drop=True)
rho_y.drop(columns=['is_sig','se'],errors='ignore',inplace=True)
rho_y[['lbl1','lbl2']] = rho_y[['lbl1','lbl2']].apply(lambda x: cat_from_map(x, di_ylbl))
rho_y['y'] = cat_from_map(rho_y['y'], di_category)
rho_y = rho_y.assign(xlbl=lambda x: x['lbl1'].astype(str) + '-' + x['lbl2'].astype(str))


##################################
# --- (2) DE-BIAS THE Y-VARS --- #

print('Fitting Nadarya-Watson to de-bias')
# Merge the amino acid lengths to the y-values
df_y_length = df_aminos.set_index('mutation').join(df_y.melt(ignore_index=False).dropna(),how='inner')
# Create a dictionary to look up the y-label (category) and then the type (lbl)
di_nw = df_y_length.groupby('category')['lbl'].unique().to_dict()
di_nw = {k1:{k2:{} for k2 in v1} for k1,v1 in di_nw.items()}
holder = []
for ylabel in di_nw:
    for tipe in di_nw[ylabel].keys():
        print(f'Fitting model for {ylabel}:{tipe}')
        tmp_data = df_y_length.query('category==@ylabel & lbl==@tipe').copy()
        x = tmp_data['length'].values
        y = tmp_data['value'].values
        di_nw[ylabel][tipe] = NadarayaWatson()
        di_nw[ylabel][tipe].fit(X=x, y=y)
        tmp_fit = di_nw[ylabel][tipe].predict(x).assign(y=y,x=x,category=ylabel, lbl=tipe)
        tmp_fit.index = tmp_data.index
        # Calculate LOO performance
        h_star = di_nw[ylabel][tipe].h
        for i, mutation in enumerate(tmp_data.index):
            y_train_i, x_train_i = np.delete(y,i),np.delete(x,i)
            tmp_mdl_i = NadarayaWatson(bw=h_star)
            tmp_mdl_i.fit(x_train_i, y_train_i)
            tmp_pred_i = tmp_mdl_i.predict(x[[i]])
            tmp_fit.loc[mutation,tmp_pred_i.columns] = tmp_pred_i.values.flatten()
        # Store result
        holder.append(tmp_fit)
# Merge
res_nw = pd.concat(holder).reset_index()
assert res_nw.shape[0] == df_y_length.shape[0]

# Calculate the performance
cn_gg = ['category','lbl']
perf_nw = bootstrap_function(res_nw, get_perf_msrs, 'value', cn_gg, n_boot, alpha, {'cn_gg':cn_gg, 'cn_y':'y', 'cn_yhat':'yhat'})
perf_nw = perf_nw.merge(res_nw.groupby(cn_gg)[['x','y']].quantile(0.2).reset_index())
perf_nw = perf_nw.pivot(cn_gg+['x','y'],'msr',['value','se'])
# Tidy up for label
perf_nw = (perf_nw*100).round(1).astype(str)
perf_nw = perf_nw.xs('r2',1,1).assign(txt=lambda x: 'R2='+x['value'] + '% (±' + x['se'] + ')')
perf_nw.drop(columns=['value','se'],inplace=True)
perf_nw.reset_index(inplace=True)

# Calculated "adjusted" y-value
adj_y = res_nw.assign(err=lambda x: x['y']-x['yhat']).pivot('mutation',['category','lbl'],['err','yhat','y'])
adj_y.to_csv(os.path.join(dir_data,'y_adjusted.csv'),index=True)


########################
# --- (3) PLOTTING --- #

# (i) Plot the within-y label correlation
posd = pn.position_dodge(0.5)
ylb = np.round(rho_y.lb.min() * 10) / 10
ax_ylb = min(0,ylb)
txt_rho_y = rho_y.groupby(['xlbl'])['rho'].mean().dropna().reset_index().assign(y=ylb-0.05)
gg_rho_y = (pn.ggplot(rho_y, pn.aes(x='xlbl',y='rho',color='y',shape='method')) + 
    pn.theme_bw() + pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.labs(y='Correlation coefficient') +
    pn.geom_text(pn.aes(y='y',label='100*rho',x='xlbl'),inherit_aes=False,size=10,format_string='{:.0f}%',data=txt_rho_y) + 
    pn.scale_color_discrete(name='Label') + 
    pn.scale_shape_discrete(name='Correlation method') + 
    pn.scale_y_continuous(limits=[ax_ylb, 1],labels=percent_format()) + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.theme(axis_text_x=pn.element_text(angle=90),axis_title_x=pn.element_blank()) + 
    pn.ggtitle('Correlation within types\nLine range shows 95% BS-CI\nText shows average correlation'))
gg_rho_y.save(os.path.join(dir_figures, 'within_y_rho.png'),width=7, height=4)


# (ii) Plot the NW-estimator
dat_nw = res_nw.copy()
dat_nw['category'] = cat_from_map(dat_nw['category'], di_category)
dat_nw['lbl'] = cat_from_map(dat_nw['lbl'], di_ylbl)
dat_nw = dat_nw.assign(grp=lambda x: x['category'].cat.codes.astype(str)+x['lbl'].cat.codes.astype(str))
perf_nw['category'] = cat_from_map(perf_nw['category'], di_category)
perf_nw['lbl'] = cat_from_map(perf_nw['lbl'], di_ylbl)
h = 2.5*dat_nw['category'].nunique()
gg_nw_ylbl = (pn.ggplot(dat_nw,pn.aes(x='x',y='y')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.geom_line(pn.aes(y='yhat',group='grp'),color='blue') + 
    pn.geom_ribbon(pn.aes(ymin='yhat-1.96*se',ymax='yhat+1.96*se'),fill='blue',alpha=0.25) + 
    pn.geom_text(pn.aes(label='txt'),data=perf_nw,size=10,nudge_y=-10,color='blue') + 
    pn.ggtitle('Dots show actual value\nBlue line and ranges is Nadarya-Watson Estimator and 95% CI for LOO-CV') + 
    pn.labs(x='Amino acid length', y='Phenotype value') + 
    pn.facet_grid('category ~ lbl',scales='free'))
gg_nw_ylbl.save(os.path.join(dir_figures,'nw_ylbl.png'),width=10, height=h)



print('~~~ End of 8_debias_y.py ~~~')