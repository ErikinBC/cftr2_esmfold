"""
This script runs the ESMFold model on a cloud compute instance (see lambda_cloud_setup.sh). The embeddings will be stored in the data/esmfold/{mutation_name}.npy file as a dictionary for the three different embedding types: ['s_s','s_z','states']. A runtime estimate will also be stored (data/runtimes.csv).
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--chunk_size', type=int, default=64, help='Chunk size for ESMFold (default: 64) - smaller value is slower but uses less memory')
parser.add_argument('--num_recycles', type=int, default=1, help='Number of recycles for ESMFold (default: 1) - larger value is accurate but slower')
parser.add_argument('--chain_linker', type=int, default=25, help='Number of residues in the linker between chains (default: 25)')
parser.add_argument('--fp_precision', type=int, default=16, help='Floating point precision (default: float16)')
parser.add_argument('--min_amino_acids', type=int, default=100, help='Minimum number of amino acids in the sequence (default: 100)')
args = parser.parse_args()
set_chunk_size = args.chunk_size
num_recycles = args.num_recycles
chain_linker = args.chain_linker
fp_precision = args.fp_precision
min_amino_acids = args.min_amino_acids

# External modules
import os
import time
import torch
import numpy as np
import pandas as pd
di_precision = {16:np.float16, 32:np.float32, 64:np.float64}

if fp_precision in di_precision:
  fp_precision = di_precision[fp_precision]
else:
  raise ValueError(f"fp_precision must be one of {list(di_precision.keys())}")

# Internal modules
from parameters import dir_data, dir_esmfold


####################################
# ---- (1) LOAD DATA & MODELS ---- #

# Load the model if it has not already been done
if "model" not in dir():
  model = torch.load("esmfold.model")
  model.eval().cuda().requires_grad_(False)

# Set the chunk size
model.set_chunk_size(set_chunk_size)

# Load the sequence data
df_aminos = pd.read_csv(os.path.join(dir_data, 'cftr_polypeptides.csv'))
# Split on the first stop codon
df_aminos['residue'] = df_aminos['residue'].apply(lambda x: x.split('_')[0],1)
df_aminos.insert(1, 'length', df_aminos['residue'].apply(len))
idx_keep = df_aminos['length'] >= min_amino_acids
print(f"Number of sequences: {idx_keep.sum()}/{len(idx_keep)}")
# Check that the amino acids are on the 20 standard amino acids
assert df_aminos['residue'].apply(list).explode().value_counts().index.isin(list('ACDEFGHIKLMNPQRSTVWY')).all(), "Not all amino acids are on the 20 standard amino acids"
df_aminos['has_len'] = idx_keep
# Split into different chunks and then sort so that we get an accurate representation of the runtime interpolation
df_aminos['group'] = pd.cut(df_aminos['length'],[0,500,750,1000,1250,1500])
df_aminos = df_aminos.sort_values(['group','mutation'],ascending=[True,False]).assign(idx=lambda x: x.groupby('group').cumcount()).sort_values(['idx','length']).reset_index(drop=True)
# Remove the "duplicated" sequences
idx_drop = df_aminos['residue'].duplicated()
print(f"Number of sequences after removing duplicates: {idx_drop.sum()}/{len(idx_drop)}")
df_aminos['not_wt'] = ~idx_drop
# Save to later transparency
df_aminos.to_csv(os.path.join(dir_data, 'dat_aminos.csv'),index=False)

# Subset to valid sets
df_aminos = df_aminos.query('not_wt & has_len').drop(columns=['not_wt','has_len']).reset_index(drop=True)

# Print approximate run time based on preliminary trials
eta_hours = df_aminos['group'].cat.codes.map({0:0.6, 1:13.9, 2:46.0, 3:135.7, 4:261.4}).sum() / 60 / 60
print(f"Approximate run time: {eta_hours:.1f} hours")
eta_time = pd.Timestamp.now() + pd.Timedelta(hours=eta_hours)
# Print the completion time in AM/PM
print(f"Approximate completion time: {eta_time.strftime('%B-%d, %I:%M %p')}")


###############################
# ---- (2) RUN SEQUENCES ---- #

holder = []
for i, r in df_aminos.iterrows():
  print(f"--- Mutation {r['mutation']} (iteration {i+1}/{len(df_aminos)}) ---")
  seq_inf = [r['residue']]

  # Run ESMFold
  stime = time.time()
  torch.cuda.empty_cache()
  output = model.infer(seq_inf, num_recycles=num_recycles, chain_linker="X"*chain_linker, residue_index_offset=512)
  # Print the inference time
  dtime = time.time()-stime
  print(f"Time taken: {dtime:.1f} seconds for a {len(seq_inf[0])} length chain \n")
  # Save the runtime
  holder.append({'mutation':r['mutation'], 'runtime':dtime, 'length':len(seq_inf[0])})

  # Convert to numpy arrays and save
  numpy_di = {k:v.detach().cpu().numpy().astype(fp_precision) for k,v in output.items() if k in ['s_s','s_z','states']}
  path = os.path.join(dir_esmfold, f"{r['mutation']}.npy")
  np.save(path, numpy_di)
# Save the runtime
df_runtime = pd.DataFrame(holder)
path_runtime = os.path.join(dir_data, 'runtimes.csv')
df_runtime.to_csv(path_runtime, index=False)

  
print('~~~ End of 5_esm_fold.py ~~~')