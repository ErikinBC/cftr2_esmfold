"""
Using the CFTR2 variants that were downloaded from 1_scrape_cftr2.py, we want to search google to get a list of NCBI links for each of these variants. For repeated 427 errors, consider:  tinyurl.com/bd73terc 

The script scrapes Google search results for genetic variants of the CFTR gene, and saves the HTML pages as files for each variant. It then extracts the NCBI links from the saved HTML pages for each variant and saves them in a CSV file.

The files saved in this script are:
1. cftr2_variants.csv: A CSV file containing the list of unique genetic variants of the CFTR gene.
2. HTML pages for each variant, saved in the data/google directory with the name of the variant as the file name.
3. ncbi_links.csv: A CSV file containing the NCBI links for each variant. This file is saved in the data directory.
"""

# External modules
import os
import sys
import requests
import numpy as np
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
# Internal modules
from parameters import dir_data, dir_google
from utilities.utils import merge_frame_with_existing

# Set up the search headres
cookies = {}
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/109.0'}


#########################
# --- (1) LOAD DATA --- #

# Load cftr_uni.csv
path_uni = os.path.join(dir_data, 'cftr2_uni.csv')
dat_uni = pd.read_csv(path_uni)

# Get the different variants
dat_variants = dat_uni.groupby(['mutation','cDNA_name','protein_name','legacy_name','num_alleles','allele_freq','pct_PI']).size().reset_index().drop(columns=[0])
assert not dat_variants['mutation'].duplicated().any(), 'There are duplicated mutations'
dat_variants.to_csv(os.path.join(dir_google, 'cftr2_variants.csv'), index=False)


################################
# --- (2) GET GOOGLE PAGES --- #

files_google = pd.Series(os.listdir(dir_google),dtype=str)
# Check if we have already searched for some of the variants
remaining_index = dat_variants[~dat_variants['mutation'].isin(files_google.str.replace('.html','',regex=False))].index
n_left = len(remaining_index)
print(f"Searching for {n_left} variants on google")

# Loop over each variant, firch search the cDNA_name, then the legacy name, and then the protein name
for i, row in dat_variants.loc[remaining_index].reset_index(drop=True).iterrows():
    print(f"Searching for {row['mutation']} ({i+1}/{n_left})")    
    # Define the search query
    part1 = f"NM_000492.4(CFTR):{row['cDNA_name']}"
    part2 = row['legacy_name']
    if row['protein_name'] != 'No Protein name':
        part3 = row['protein_name']
    else:
        part3 = ''
    # Search for the variant on google
    search_query = f"{part1} OR {part2} OR {part3}"
    search_url = f"https://www.google.com/search?q={search_query}"
    if len(cookies) == 0:
        search_res = requests.get(search_url, headers=headers)
    else:
        search_res = requests.get(search_url, headers=headers, cookies=cookies)
    if search_res.status_code == 200:
        cookies = search_res.cookies.get_dict()
    if search_res.status_code == 429:
        sys.exit('Too many requests, shutting down')
    # Save the google page
    path_google = os.path.join(dir_google, f"{row['mutation']}.html")
    with open(path_google, 'w') as f:
        f.write(search_res.text)
    # Pause for 30-60 random seconds
    nsleep = np.random.randint(30,180)
    print(f"Sleeping for {nsleep} seconds")
    sleep(nsleep)


##############################
# --- (3) GET NCBI LINKS --- #

# Set the path to store the NCBI links
path_ncbi = os.path.join(dir_data, 'ncbi_links.csv')

# Store the different link possiblities
ncbi_links = {}
for i, row in dat_variants.reset_index(drop=True).iterrows():
    print(f"Searching for {row['mutation']} ({i+1}/{len(dat_variants)})")
    # Load the google page
    path_google = os.path.join(dir_google, f"{row['mutation']}.html")
    assert os.path.exists(path_google), f"Could not find {path_google}"
    with open(path_google, 'r') as f:
        search_res = f.read()
    if len(search_res) < 5000:
        sys.exit('Woops! Looks like google deteced suspicious activity')
    # Extract the links
    soup_res = BeautifulSoup(search_res, 'html.parser')
    link_res = pd.Series([a.attrs['href'] for a in soup_res.findAll('a', href=True)])
    link_res = link_res[link_res.str.contains('ncbi',na=False,case=False)].drop_duplicates()
    # Store
    assert len(link_res) > 0, "Could not find {row['mutation']}"
    ncbi_links[row['mutation']] = link_res.to_list()

# Save data
dat_ncbi = pd.DataFrame.from_dict(ncbi_links, orient='index')
dat_ncbi = dat_ncbi.melt(ignore_index=False,var_name='num',value_name='link').rename_axis('mutation').dropna().sort_values('mutation').reset_index()
dat_ncbi = merge_frame_with_existing(df=dat_ncbi, path=path_ncbi)
dat_ncbi.to_csv(path_ncbi, index=False)

print('~~~ End of 2_get_ncbi.py ~~~')