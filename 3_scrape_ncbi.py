"""
This script extracts relevant information from NCBI links that were obtained from the 2_get_ncbi.py script (e.g. genomic location, submission count, mutation name) by downloading each of these pages to a folder and parsing the HTML. The final output will be stored as ncbi_genome_loc.csv.

The following files are saved:

1. HTML pages from the NCBI website containing information about the gene mutations are downloaded and stored in the folder ncbi. For each HTML page, a file is created in the ncbi folder with the name '[ID].html', where [ID] is the ID associated with the corresponding NCBI page.
2. dat_href.csv: For each of the NCBI pages from generated from the ncbi_links.csv page, extracts the chromosome location information
3. ncbi_genome_loc.csv: final output file that contains the relevant information extracted from the HTMLs including the location of the genome where the mutation occurs.
"""

# External modules
import os
import numpy as np
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from Levenshtein import distance as levenshtein_distance
# Internal modules
from utilities.utils import str2list
from parameters import dir_data, dir_ncbi

# Load cftr_uni.csv
path_uni = os.path.join(dir_data, 'cftr2_uni.csv')
dat_uni = pd.read_csv(path_uni)
dat_mut_names = dat_uni.groupby(['mutation','cDNA_name','protein_name']).size().reset_index().drop(columns=[0])


#########################
# --- (1) LOAD URLs --- #

# Load the URLs
url_path = os.path.join(dir_data, 'ncbi_links.csv')
df_url = pd.read_csv(url_path)
# Needs to be in the format of www.ncbi.
df_url = df_url[df_url['link'].str.contains('www\\.ncbi\\.',regex=True)].reset_index(drop=True)
# Remove the ClearHT command
df_url['link'] = df_url['link'].str.replace('?cmd=ClearHT&', '', regex=False)
# Remove trailing forward slash
df_url['link'] = df_url['link'].str.replace('\\/$', '', regex=True)
# Get a URL ID
df_url['id'] = df_url['link'].str.split('\\/').apply(lambda x: x[-1],1)
# Check that there is a 1:1 mapping between link and ID
assert df_url.groupby('link')['id'].nunique().max() == 1, 'There is not a 1:1 mapping between link and ID'
assert df_url.groupby('id')['link'].nunique().max() == 1, 'There is not a 1:1 mapping between link and ID'
# Note that some id's will be duplicated because the same link shows up multiple times

# Create a dataframe that allows a mapping between the id column, and the associated mutation and cDNA name
mapping_id_mut_cDNA = df_url[['mutation','id']].merge(dat_mut_names,'left')


##############################
# --- (2) DOWNLOAD HTMLs --- #

# Get the unique link/id
dat_links = df_url.groupby(['link','id']).size().reset_index().drop(columns=[0])
# Find which URLs have already been downloaded
ncbi_files = pd.Series(os.listdir(dir_ncbi), dtype=str)
ncbi_ids = ncbi_files.str.replace('.html', '', regex=False)
# Filter the URLs
remaining_index = dat_links[~dat_links['id'].isin(ncbi_ids)].index
n_left = len(remaining_index)
print(f'Downloading {n_left} NCBI pages')
# Create a dictionary mapping id's to links
id2link = dict(zip(dat_links['id'], dat_links['link']))

# Download the pages
for i, row in dat_links.loc[remaining_index].iterrows():
    # Download the page to the ncbi folder
    link, ncbi_id = row['link'], row['id']
    ncbi_file = os.path.join(dir_ncbi, '{}.html'.format(ncbi_id))
    if not os.path.exists(ncbi_file):
        os.system('wget -O {} {}'.format(ncbi_file, link))
        # Wait for 1 second to avoid overloading the server
        sleep(1) 


################################
# --- (3) EXTRACT KEY INFO --- #

# We are looking for the following information:
# (i) genomic location
# (ii) submission count
# (iii) mutation name

holder = []
for i, row in dat_links.iterrows():
    print('Processing {} of {}'.format(i+1, dat_links.shape[0]))
    # Load the HTML file
    ncbi_file = f"{dir_ncbi}/{row['id'] }.html"
    assert os.path.exists(ncbi_file), 'File does not exist: {}'.format(ncbi_file)
    with open(ncbi_file, 'r') as f:
        txt = f.read()
    # Remove excess whitespace
    txt = ' '.join(txt.split())
    
    # --- (i) genomic location --- #
    idx_loc = txt.lower().find('genomic location')
    if idx_loc >= 0:
        soup = BeautifulSoup(txt[idx_loc:idx_loc+1000], features='lxml')
        href_loc = soup.find('a').get('href')
    else:
        print('Cannot find genomic location for {}'.format(row['link']))
        continue

    # --- (ii) submission count --- #    
    soup = BeautifulSoup(txt, features='xml')
    dd_sub = [d for d in soup.findAll('dd') if 'submission' in d.text]
    n_dd_sub = len(dd_sub)
    assert n_dd_sub in [0, 1], 'Multiple submission counts found'
    if n_dd_sub == 1:
        n_sub = int(dd_sub[0].text.split(' ')[0])
    else:
        # If we cannot find submission in a <dd> bracket, check for <dt>
        idx_dt = txt.find('<dt>Submissions:</dt>')
        assert idx_dt >= 0, f"Cannot find submission count for {row['link']} ({row['id']})"
        n_sub = int(''.join(BeautifulSoup(txt[idx_dt-100:idx_dt+100], features='lxml').find('dd').text.split(' ')))
    
    # --- (iii) mutation name --- #
    if n_dd_sub == 1:
        idx_name = txt.find('<dt>Preferred name:</dt>')
        assert idx_name >= 0, 'Cannot find preferred name'
        mut_names = [d.text for d in BeautifulSoup(txt[idx_name:idx_name+500], features='lxml').findAll('dd')]
    else:
        mut_names = str2list(soup.find('h2').text)
    assert len(mut_names) > 0, 'No mutation names found'

    # Store for later
    res = pd.DataFrame({'id':row['id'], 'href':href_loc, 'n_sub':n_sub, 'names':[mut_names]}, index=[i])
    holder.append(res)

# Merge
dat_href = pd.concat(holder).explode('names').reset_index(drop=True)
# Save for later
dat_href.to_csv(os.path.join(dir_data, 'dat_href.csv'), index=False)


#############################
# --- (4) CLEAN UP DATA --- #

# --- (i) Clean up the genomic location (href) --- #
remove_strings = ['https://www.ncbi.nlm.nih.gov/variation/view/?',
                  '/variation/view/?']
for s in remove_strings:
    dat_href['href'] = dat_href['href'].str.replace(s, '', regex=False)
# Se shoould expect 5 columns afterwards
dat_href['href'] = dat_href['href'].str.split('\\&')
idx_drop = dat_href['href'].apply(len,1) != 5
print('Dropping {} rows because they do not have 5 columns'.format(idx_drop.sum()))
dat_href = dat_href[~idx_drop].reset_index(drop=True)
# Look for the following information: chr, from, to, and assm
dat_href = dat_href.explode('href').reset_index(drop=True)
tmp = dat_href['href'].str.split('\\=',n=1,expand=True)
tmp.columns = ['key','val']
# Concatenate together
dat_href = pd.concat(objs=[dat_href, tmp], axis=1).drop(columns=['href'])
cn_idx = np.setdiff1d(dat_href.columns, ['key','val'])
cn_keep = ['assm', 'chr', 'from', 'to']
dat_href = dat_href.pivot(cn_idx, 'key','val')[cn_keep].reset_index()
# Check chromosome
dat_href['chr'] = dat_href['chr'].astype(int)
assert (dat_href['chr'] == 7).all(), 'CFTR is should be on chromosome 7'
dat_href.drop(columns=['chr'], inplace=True)
# Check assembly version
assert (dat_href['assm'].str.split('\\.',n=1,expand=True)[0] == 'GCF_000001405').all(), 'Assembly should be GRCh38 (patches ignored)'
dat_href.drop(columns=['assm'], inplace=True)
# Set the genomic coordinates to integers
dat_href[['from','to']] = dat_href[['from','to']].astype(int)

# --- (ii) Clean up search names --- #
# Remove parantheses
dat_href['names'] = dat_href['names'].str.replace('\\(|\\)', '', regex=True)
dat_href['names'] = dat_href['names'].str.strip()
# Check that we have at least one cDNA name per id
assert dat_href.assign(check=lambda x: x['names'].str.contains('c\\.',regex=True)).groupby('id')['check'].any().all(), 'No cDNA name found'
# Hive off the cDNA and amino acid change
dat_href = dat_href.assign(names=lambda x: x['names'].str.split('\\:|\\s|\\[|N[MGCP]\\_',regex=True)).explode('names')
dat_href = dat_href[dat_href['names'].str.contains('c\\.|p\\.',regex=True,na=False)]
dat_href = dat_href.drop_duplicates().reset_index(drop=True)
# Remove the c. and p.-only prefixes
dat_href = dat_href[~dat_href['names'].isin(['c.','p.'])]
# Check that every mutation has at least one id
tmp = mapping_id_mut_cDNA[['mutation','id']].merge(dat_href.groupby('id').size().reset_index().rename(columns={0:'n'}),'left').assign(check=lambda x: x['n'].notnull()).groupby('mutation')['check'].any().reset_index()
missing_muts = tmp[~tmp['check']]['mutation'].to_list()
print(f"The following {len(missing_muts)} mutations are missing: {', '.join(missing_muts)}")


# --- (iii) Clean up search names --- #
name_match = mapping_id_mut_cDNA.merge(dat_href, on='id')
# For each mutation, find the cDNA and protein_name that has the highest match
name_match = name_match.melt(['mutation','id','from','to','names'],['cDNA_name','protein_name'],var_name='name_type',value_name='val')
name_match['name_type'] = name_match['name_type'].str.replace('_name','',regex=False)
# Remove the "No protein name"
name_match = name_match[~((name_match['name_type'] == 'protein') & (name_match['val'] == 'No protein name'))]
# Remove trailing
name_match['val'] = name_match['val'].str.strip()
# Check the prefex of each and subset to c/p only
idx1 = name_match['names'].str.split('\\.',1,True)[0].isin(['c','p'])
idx2 = name_match['val'].str.split('\\.',1,True)[0].isin(['c','p'])
name_match = name_match[idx1 & idx2]
# Check that the first two characters match
name_match = name_match[name_match['names'].str[:2] == name_match['val'].str[:2]].reset_index(drop=True)
# Find the levenshen distance for the columns 'val' and 'names'
name_match = name_match.assign(lev=lambda x: x.apply(lambda y: levenshtein_distance(y['val'],y['names']),axis=1))
# For each mutation and name_type, find the id that has the lowest levenshtein distance
name_match_min = name_match.loc[name_match.groupby(['mutation','name_type'])['lev'].idxmin()].copy()
name_match_min = name_match_min.sort_values(['mutation','name_type']).reset_index(drop=True)
assert name_match_min.groupby('mutation').size().isin([1,2]).all(), 'Each mutation should have one cDNA and possibly a protein name'
# Define the genomic position as to/from
name_match_min['tofrom'] = name_match_min[['to','from']].sum(1)
# Determine which mutations have a disagreement
name_match_min = name_match_min.merge((name_match_min.groupby('mutation')['tofrom'].nunique()>1).reset_index().rename(columns={'tofrom':'disagree'}))
# If the smallest levenshtein distance is >= 10, then we will drop this mutation
tmp = name_match_min.groupby('mutation')['lev'].min()
name_match_min = name_match_min[~name_match_min['mutation'].isin(tmp[tmp >= 10].index)]
# If there is still disagreement, we will pick the complementary DNA name if the numbers align, otherwise we will pick the protein
tmp1 = name_match_min[~name_match_min['disagree']].copy()
tmp2 = name_match_min[name_match_min['disagree']].copy()
# First, calculate whether the cDNA aligns
tmp3 = tmp2[tmp2['name_type']=='cDNA'].set_index('mutation')[['names','val']]
tmp4 = tmp3.apply(lambda x: x.str.replace('c\\.|\\_','',regex=True).str.split('[^0-9]',n=1,expand=True)[0],axis=1)
tmp5 = tmp4.assign(use_c=lambda x: x['names'] == x['val'])[['use_c']].reset_index()
# Next, calculate whether the protein aligns
tmp6 = tmp2[tmp2['name_type']=='protein'].set_index('mutation')[['names','val']]
tmp7 = tmp6.apply(lambda x: x.str.replace('p.','',regex=False).str.split('[0-9]',n=1,expand=True)[0],axis=1)
tmp7 = tmp7.assign(use_p=lambda x: x['names'] == x['val'])[['use_p']].reset_index()
# Merge together an pick the ones that align
tmp8 = tmp2.merge(tmp5).merge(tmp7)
tmp9 = tmp8.assign(keep=lambda x: (x['use_c'] & (x['name_type']=='cDNA')) | (~x['use_c'] & x['use_p'] & (x['name_type']=='protein')))
tmp9 = tmp9[tmp9['keep']]

# Combine the two dataframes, picking the location that aligns
ncbi_genome_loc = pd.concat(objs=[tmp1,tmp9],axis=0).pivot(['mutation','to','from'],'name_type',['names','id'])
ncbi_genome_loc['names'].fillna('no_ncbi',inplace=True)
ncbi_genome_loc.rename(columns={'names':'ncbi','id':'link'},inplace=True)
ncbi_genome_loc = ncbi_genome_loc.swaplevel(0,1,axis=1)
# Rename the columns
ncbi_genome_loc.columns = ncbi_genome_loc.columns.map('_'.join)
# Replace the 'id' with a url link
cn_links = ['cDNA_link','protein_link']
ncbi_genome_loc[cn_links] = ncbi_genome_loc[cn_links].replace(id2link)
ncbi_genome_loc[cn_links] = ncbi_genome_loc[cn_links].mask(~(ncbi_genome_loc[cn_links].apply(lambda x: x.str[:4]) == 'http'))
ncbi_genome_loc[cn_links] = ncbi_genome_loc[cn_links].fillna('')
ncbi_genome_loc['links'] = ncbi_genome_loc[cn_links].apply(lambda x: ','.join(list(x.unique())),axis=1)
ncbi_genome_loc.drop(columns=cn_links,inplace=True)
# Add on the names from the cftr2 table
tmp = dat_mut_names.set_index('mutation')
tmp.columns = 'cftr2_' + tmp.columns.str.replace('_name','',regex=False)
tmp = tmp.apply(lambda x: x.str.strip())
ncbi_genome_loc = ncbi_genome_loc.join(tmp).reset_index()
# Put links as the last column
ncbi_genome_loc = ncbi_genome_loc.drop(columns='links').assign(links=ncbi_genome_loc['links'])
# Save the data
ncbi_genome_loc.to_csv(os.path.join(dir_data,'ncbi_genome_loc.csv'),index=False)


print('~~~ End of 3_scrape_ncbi.py ~~~')