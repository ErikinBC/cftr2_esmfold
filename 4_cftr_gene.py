"""
The script loads in the genomic coordinates obtained from the NCBI page and then applies the actual mutation to get the new amino acid sequence (see data/cftr_polypeptides.csv). A copy of the exonic locations is also stored (data/cftr_exon_locs.csv).

---------------
Reference URLs: 

SickKids gene:
http://www.genet.sickkids.on.ca/MRnaPolypeptideSequencePage,form0.direct

Ensembl gene:
https://useast.ensembl.org/Homo_sapiens/Transcript/Summary?db=core;g=ENSG00000001626;r=7:117480025-117668665;t=ENST00000003084
"""

# Add an argument for the ensemble version number
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ensembl_version', type=int, default=109, help='Ensembl version number')
args = parser.parse_args()
ensembl_version = args.ensembl_version

# External modules
import os
import re
import requests
import numpy as np
import pandas as pd
import plotnine as pn
from pyensembl import EnsemblRelease
# Internal modules
from parameters import dir_figures, dir_data
from utilities.utils import translate_dna, get_cDNA_variant_types


#################################
# --- (1) LOAD GENOMIC LOCS --- #

# (i) Load the genomic data location
ncbi_genome_loc = pd.read_csv(os.path.join(dir_data, 'ncbi_genome_loc.csv'))
ncbi_genome_loc = ncbi_genome_loc[['mutation','cftr2_cDNA','from','to']].rename(columns={'cftr2_cDNA':'cDNA'})
# Add on types
ncbi_genome_loc['vartype'] = get_cDNA_variant_types(ncbi_genome_loc['cDNA'])
print(f"Dropping {ncbi_genome_loc['vartype'].isnull().sum()} rows with no variant type")
ncbi_genome_loc = ncbi_genome_loc[~ncbi_genome_loc['vartype'].isnull()].reset_index(drop=True)


############################
# --- (2) LOAD SK CFTR --- #

# Load the CFTR gene sequence from SK
url_sk = 'http://www.genet.sickkids.on.ca/cftrdnasequence/cftrdnasequence-427_1573.txt?endPoint=500000&startPoint=0'
r = requests.get(url_sk)
# Loop through and find all the exons...
sk_bases = r.text.replace('\n','').replace(' ','')
print(f"SK CFTR gene sequence is {len(sk_bases)} bases long")
sk_exons = pd.Series(sk_bases).str.replace('[a-z]+','-',regex=True).str.split('-',regex=False).explode().reset_index(drop=True).rename_axis('exon').reset_index()
sk_exons = sk_exons.rename(columns={0:'bases'}).assign(exon=lambda x: x['exon']+1)
sk_exons['n'] = sk_exons['bases'].str.len()
sk_exons = sk_exons[sk_exons['n'] > 0]
print(f"SK CFTR gene sequence has {sk_exons['exon'].nunique()} exons and a length of {sk_exons['n'].sum()} bases")


#################################
# --- (3) LOAD ENSEMBL CFTR --- #

# (i) Load the CFTR gene sequence from GRCh38
data = EnsemblRelease(ensembl_version)
cftr_gene = data.genes_by_name('CFTR')[0]
assert cftr_gene.genome.reference_name == 'GRCh38', "The genome reference name is not GRCh38"
print(f"CFTR gene sequence is {cftr_gene.end - cftr_gene.start} bases long (from {cftr_gene.start} to {cftr_gene.end}))")
# Get the exons as a dataframe
cftr_exons = pd.concat([pd.DataFrame(e.to_dict(),index=[0]) for e in cftr_gene.exons]).reset_index(drop=True)
print(f"CFTR gene sequence has {len(cftr_exons)} exons")
cn_keep = cftr_exons.nunique() > 1
cftr_exons = cftr_exons.loc[:,cn_keep]
# Calculate the exon length
cftr_exons['exon_length'] = cftr_exons['end'] - cftr_exons['start']
# Find the "canonical" transcript which aligns with the protein
di_tid = [data.transcript_by_id(tid) for tid in data.transcript_ids_of_gene_id(cftr_gene.gene_id)]
canonical_tid = [tid for tid in di_tid if tid.complete and tid.support_level is not None]
assert len(canonical_tid) == 1, "There is not exactly one canonical transcript"
canonical_tid = canonical_tid[0]
# Check range
assert canonical_tid.start >= cftr_gene.start, 'Canonical transcript start is before the gene start'
assert canonical_tid.end <= cftr_gene.end, 'Canonical transcript end is after the gene end'

# (ii) Get coding sequence range (i.e. the exons less 5' and 3' UTR)
canonical_exons = pd.DataFrame(canonical_tid.coding_sequence_position_ranges).rename(columns={0:'start',1:'end'}).rename_axis('exon').reset_index().assign(exon=lambda x: x['exon']+1, n=lambda x: x['end'] - x['start'] + 1)
# The exonic range does not include stop codon at the end
assert canonical_exons['n'].sum() == len(canonical_tid.coding_sequence) - 3, 'The exonic range does not include stop codon at the end'
# Add an extra exons for the 3' and the end
n_exons = len(canonical_exons)
canonical_exons.loc[n_exons-1, 'end'] += 3
canonical_exons.loc[n_exons-1, 'n'] += 3

# Add on the bases
ref_loc = canonical_exons[['start','end']].subtract(canonical_exons['start'], axis=0)
ref_loc = ref_loc.assign(end=lambda x: x['end']+1).assign(end=lambda x: x['end'].cumsum()).assign(start=lambda x: x['end'].shift(1).fillna(0).astype(int))
ref_loc = ref_loc.apply(lambda x: canonical_tid.coding_sequence[x['start']:x['end']],axis=1)
assert (ref_loc.apply(lambda x: len(x), 1) == canonical_exons['n']).all(), 'The reference location does not match the exon length'
canonical_exons['bases'] = ref_loc.copy()
print(f"Canonical CFTR gene sequence has {len(canonical_exons)} exons and a length of {canonical_exons['n'].sum()} bases")


##############################
# --- (4) COMPARE EXONS --- #

# (i) Check for Ensembl and SK differences
cn_comp = ['exon', 'bases', 'n']
dat_comp_exon = sk_exons[cn_comp].merge(canonical_exons[cn_comp], on='exon', suffixes=('_sk','_canonical')).melt('exon',var_name='tmp')
tmp = dat_comp_exon['tmp'].str.split('_',expand=True,n=1).rename(columns={0:'msr',1:'source'})
dat_comp_exon = pd.concat([dat_comp_exon,tmp],axis=1).drop('tmp',axis=1)
dat_comp_exon = dat_comp_exon.pivot(['msr','exon'],'source','value')

# (ii) Aggregate alignment between exons
pct_exon_n = dat_comp_exon.loc['n'].assign(check=lambda x: x['sk'] == x['canonical'])['check'].mean()
print(f"Percent of exons with the same number of bases: {pct_exon_n:.3f}")
assert pct_exon_n == 1, "The number of bases in the exons is not the same"
pct_exon_dna = dat_comp_exon.loc['bases'].assign(check=lambda x: x['sk'] == x['canonical'])['check'].mean()
print(f"Percent of exons with same DNA bases: {pct_exon_dna:.3f}")

# (iii) Calculate the position and spot check...
cn_idx = ['exon','start','end']
bases_comp = canonical_exons[cn_idx].merge(dat_comp_exon.loc['bases'].reset_index()).melt(cn_idx,var_name='source',value_name='seq')
bases_comp = bases_comp.assign(seq=lambda x: x['seq'].apply(list)).explode('seq').assign(idx=lambda x: x.groupby(['exon','source'])['seq'].cumcount()+x['start'])
# The last idx should be the end of the sequence
assert bases_comp.groupby(['exon','source']).agg({'end':'max','idx':'max'}).reset_index().assign(check=lambda x: x['end'] == x['idx'])['check'].all(), 'The last idx should be the end of the sequence'
bases_comp = bases_comp.pivot(['exon','idx'],'source','seq')
bases_comp['aligns'] = bases_comp.apply(lambda x: x['sk'] == x['canonical'], axis=1)
nonaligned_bases = bases_comp[~bases_comp['aligns']].drop(columns='aligns')
print(f"There are {len(nonaligned_bases)} bases that do not align between the SK and Ensembl sequences")
# Note, canonical aligns with UCSC Genome Browser on Human (GRCh38/hg38)
url_ucsc = 'https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr7;start=%s;end=%s'
for i, r in nonaligned_bases.reset_index().iterrows():
    url_query = url_ucsc % (r['idx']-1, r['idx'])
    res_ucsc = requests.get(url_query).json()
    assert len(res_ucsc['dna']) == 1, 'The UCSC query should return a single base'
    assert r['canonical'] == res_ucsc['dna'], 'The UCSC query should return the same base as the canonical'


##############################
# --- (5) APPLY VARIANTS --- #

# (i) Get long form of the gene
cftr_gene = bases_comp[['canonical']].rename(columns={'canonical':'dna'}).reset_index()
# Save Exon data for later
cftr_gene.to_csv(os.path.join(dir_data, 'cftr_exon_locs.csv'), index=False)


# (ii) Loop over the different variant types
assert ncbi_genome_loc['vartype'].isin(['mutation','delete','dup','ins']).all(), 'The variant type is not recognized'
# Look for intronic mutations
intronic_muts = ncbi_genome_loc.loc[~ncbi_genome_loc['from'].isin(cftr_gene['idx']),'mutation']
print(f"There are {len(intronic_muts)} intronic mutations")
ncbi_exome_loc = ncbi_genome_loc[~ncbi_genome_loc['mutation'].isin(intronic_muts)]
# Remove mutations of uncertain base changes
ncbi_exome_loc = ncbi_exome_loc[~ncbi_exome_loc['cDNA'].str.contains(';',regex=False)].reset_index(drop=True)

# Store the amino acid sequences in a dict
di_polypeptide = {'base':translate_dna(''.join(cftr_gene['dna']))}
for i, r in ncbi_exome_loc.iterrows():
    print('Processing variant %s of %s' % (i+1, len(ncbi_exome_loc)))
    if r['vartype'] == 'mutation':
        mutation = re.split(r'[0-9]', r['cDNA'])[-1]
        mutation = re.sub('[^ACTG\\-\\>]','',mutation).split('>')
        assert mutation[0] in 'ACTG' and mutation[1] in 'ACTG', 'The mutation is not recognized'
        # Check that the reference aligns with the genome
        assert r['from'] == r['to'], 'The mutation should be a single base'
        if cftr_gene.loc[cftr_gene['idx'] == r['from'],'dna'].values[0] != mutation[0]:
            print('The mutation does not align with the genome sequence: %s (%s)' % (r['cDNA'], r['mutation']))
            continue
        dna_seq = cftr_gene.assign(dna=lambda x: np.where(x['idx'] == r['from'], mutation[1], x['dna']))['dna']
    if r['vartype'] == 'delete':
        dna_seq = cftr_gene.loc[~cftr_gene['idx'].isin(np.arange(r['from'], r['to']+1)), 'dna']
    if r['vartype'] == 'dup':  # Implies insertion
        # For example, https://www.ncbi.nlm.nih.gov/snp/rs387906360#variant_details shows that a TC insertion changes TCA -> TCTCA for c.1021_1022dupTC, and ATT -> AGATATT for c.1327_1330dupGATA
        dna2insert = r['cDNA'].split('dup')[-1]
        dna_seq = pd.Series(list(cftr_gene.loc[cftr_gene['idx'] <= r['from'],'dna']) + \
            list(dna2insert) +\
            list(cftr_gene.loc[cftr_gene['idx'] >= r['to'],'dna']))
    if r['vartype'] == 'ins':  # Implies insertion
        # For now, we will only do single base insertions
        if r['from'] + 1 == r['to']:
            # For the insertion, if the from-to is 10-11, then the insertion is between 10 and 11. For example, https://www.ncbi.nlm.nih.gov/snp/rs397508138#variant_details shows that ATA becomes AGTA, where the first A and T where the "from" and "to" position
            dna_seq = pd.Series(list(cftr_gene.loc[cftr_gene['idx'] <= r['from'],'dna']) + \
                [r['cDNA'][-1]] + \
                list(cftr_gene.loc[cftr_gene['idx'] >= r['to'],'dna']))
    residues = translate_dna(''.join(dna_seq))
    di_polypeptide[r['mutation']] = residues
# Merge and save
df_polypeptide = pd.DataFrame.from_dict(di_polypeptide,orient='index').rename_axis('mutation').rename(columns={0:'residue'}).reset_index()
df_polypeptide.to_csv(os.path.join(dir_data, 'cftr_polypeptides.csv'), index=False)
print(f'Number of protein variants: {len(df_polypeptide)-1}')

# (iii) Plot the cumulative distribution of amino acid sizes
dat_plot = df_polypeptide['residue'].str.split('_',expand=True,n=1)[0].apply(lambda x: len(x)).value_counts().sort_index().cumsum().reset_index().rename(columns={'index':'amino_acid_size',0:'cumulative_count'})
dat_plot['pct'] = dat_plot['cumulative_count'] / dat_plot['cumulative_count'].max()
gg_n_amino = (pn.ggplot(dat_plot, pn.aes(x='amino_acid_size', y='100*pct')) + 
                pn.geom_step() + pn.theme_bw() +
                pn.ggtitle(f'Amino acid length to first stop codon ({df_polypeptide.shape[0]-1} variants)') + 
                pn.labs(x='Number of amino acids', y='Cumulative proportion of variants (%)'))
gg_n_amino.save(os.path.join(dir_figures, 'fig_n_amino.png'), width=6, height=4)


print('~~~ End of 4_cftr_gene.py ~~~') 