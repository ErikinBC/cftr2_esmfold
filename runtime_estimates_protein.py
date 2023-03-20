"""
Download the combine the protein shapes

https://github.com/plotly/dash-bio
https://www.ebi.ac.uk/pdbe/entry/pdb/6o2p/protein/1
https://towardsdatascience.com/visualizing-and-analyzing-proteins-in-python-bd99521ccd

python scripts/esmfold_inference.py --fasta examples/data/few_proteins.fasta -o tmp_few_proteins --cpu-only


https://lambdalabs.com/blog/getting-started-with-lambda-cloud-gpu-instances
https://lambdalabs.com/blog/downloading-data-sets-lambda-cloud
"""

import os
import requests
import numpy as np
import pandas as pd
import plotnine as pn

# Set up folders
dir_data = os.path.join('data')
dir_figures = os.path.join('figures')

# Load the peptides
polypeps = pd.read_csv(os.path.join(dir_data, 'cftr_polypeptides.csv'))


################################
# --- (1) WILDTYPE PROTEIN --- #

# Double check FASTA Download from the protein data bank
fasta_uniprot = ''.join(requests.get('https://rest.uniprot.org/uniprotkb/P13569.fasta').text.split('\n')[1:])
assert fasta_uniprot  == polypeps.loc[polypeps['mutation'] == 'base', 'residue'].values[0].replace('_',''), 'FASTA sequence does not match the base sequence'


################################
# --- (2) ESTIMATE RUNTIME --- #

data = """100,1.2
200,4.5
300,12.2
400,27.5
500,49.4
600,89.7"""
n_train = 4
df_run = pd.DataFrame([d.split(',') for d in data.split('\n')], columns = ['length', 'time'])
df_run['length'] = df_run['length'].astype(int)
df_run['time'] = df_run['time'].astype(float)
# Fit different degrees
x_test, y_test = df_run['length'], df_run['time']
holder = []
for deg in range(1,6):
    x_train, y_train = df_run['length'].values[:n_train], df_run['time'].values[:n_train]
    mdl = np.poly1d(np.polyfit(x_train, y_train, deg=deg))
    yhat_test = mdl(x_test)
    res = pd.DataFrame({'length':x_test, 'pred':yhat_test, 'y':y_test, 'deg':deg})
    holder.append(res)

# Merge
df_poly = pd.concat(holder).reset_index(drop=True)

# Plot
p = (pn.ggplot(df_poly, pn.aes(x='length', y='pred', color='factor(deg)')) +
        pn.geom_point() + pn.theme_bw() + pn.geom_line() +
        pn.labs(x='Length of protein', y='Time (seconds)') +
        pn.geom_point(pn.aes(x='length', y='time'), color='black', size=2, data=df_run) +
        pn.geom_line(pn.aes(x='length', y='time'), color='black', size=2, data=df_run))
p.save(os.path.join(dir_figures, 'protein_runtime.png'), width=6, height=4, dpi=300)

# Get the "best" fit
df_poly['resid'] = (df_poly['y'] - df_poly['pred'])**2
deg_star = df_poly.groupby('deg')['resid'].mean().sort_values().index[0]
mdl_star = np.poly1d(np.polyfit(x_test, y_test, deg=deg))

# Make runtime predictions for the proteins
protein_lengths = np.sort(polypeps['residue'].str.split('_').apply(lambda x: x[0], 1).apply(len).unique())
# Number of minutes
protein_runtime = pd.DataFrame({'length':protein_lengths, 'minutes':mdl_star(protein_lengths) / 60})
# Plot
p = (pn.ggplot(protein_runtime, pn.aes(x='length', y='minutes')) + 
        pn.geom_point() + pn.theme_bw() + pn.geom_line() +
        pn.labs(x='Length of protein', y='Time (minutes)'))
p.save(os.path.join(dir_figures, 'protein_runtime_predictions.png'), width=6, height=4, dpi=300)


