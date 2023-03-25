"""
This script has three main sections:

1. The script starts by accepting optional arguments using the argparse module, which allows users to specify whether to scrape certain sections of the data or not. The arguments that can be passed include whether to scrape the single SNP variant data (--scrape_variants), whether to scrape the two-SNP variant data (--scrape_combinations), and the number of variant combinations to scrape for the combinations (--num_variants). The script then prints out which sections are being scraped based on the arguments that were passed.

2. In the second section, the script starts a session and navigates to the CFTR2 mutations history page on https://cftr2.org/mutations_history to download an Excel file containing information about variants. The file is saved in the data directory. The script then gets the JSON file-version of the mutations from https://cftr2.org/json/cftr2_mutations, encodes it to HTML, and creates lookup dictionaries. It then processes the downloaded Excel file by renaming columns and removing rows that have null values. Finally, it prints out any legacy names that are not in the JSON file and any JSON names that are not in the legacy names.

3. In the third section, the script downloads data for individual variants if the --scrape_variants argument was passed. It does this by looping over each mutation and extracting the data, which is then saved in a CSV file in the data directory. The name of the file is cftr2_uni.csv. No files are saved if the --scrape_variants argument was not passed.

4. In the fourth section, if the --scrape_combinations flag is passed, the mutation mutation combinations are scraped and saved as cftr2_comb.csv.
"""

# Add optional arguments about whether the scrape certain sections
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scrape_variants', action='store_true', help="Whether the single SNP variant data should be scraped (unless this flag is present, this won't be scraped)")
parser.add_argument('--scrape_combinations', action='store_true', help="Whether the two-SNP variant data should be scraped (unless this flag is present, this won't be scraped)")
parser.add_argument('--scrape_cftr1', action='store_true', help="Whether the CFTR database should be scraped to get a unique list of mutations that have ever been discovered")
parser.add_argument('--num_variants', type=int, default=25000, help='Number of variant combinations to scrape for the combinations (default==1500)')
args = parser.parse_args()
scrape_variants = args.scrape_variants
scrape_combinations = args.scrape_combinations
scrape_cftr1 = args.scrape_cftr1
num_variants = args.num_variants
print(f'Scraping variants: {scrape_variants}')
print(f'Scraping combinations: {scrape_combinations}')
print(f'Scraping CFTR1 database: {scrape_cftr1}')
print(f'Number of variants: {num_variants}')

# External modules
import os
import re
import requests
import numpy as np
import pandas as pd
from time import sleep, time
from bs4 import BeautifulSoup
from urllib.parse import quote
from itertools import combinations
# Internal modules
from parameters import dir_data
from utilities.utils import merge_html_tables, merge_frame_with_existing


##################################################
### --- (1) START SESSION AND GET VARIANTS --- ###

# To access the data, go to following URL, look for classes with "This field is required.", fill in the form, and click the "Accept" button
request_url = 'https://cftr2.org'
val = '1'
payload = {'education':val, 'individual':val, 'discuss':val, 'privacy':val, 'op':'Accept', 'form_build_id':'form-uMFXGGGPBjj_Q3K5ExFUMZoW-PsUWsj1R7ZHScXyHpM', 'form_id':'cftr2_agreement_form'}

# Start a session
s = requests.Session()
r = s.post(request_url, data=payload)
# Navigate to the variant list page
page_mutations = s.get('https://cftr2.org/mutations_history')
# Find all the xlsx links
links_mutations = pd.Series([link for link in BeautifulSoup(page_mutations.text, "html.parser").findAll('a')])
links_mutations = links_mutations[links_mutations.str.len() > 0]
links_mutations = links_mutations.astype(str).str.split('\\"',regex=True,n=2,expand=True)[1]
links_mutations = links_mutations[links_mutations.str.contains('xlsx',regex=False)]
# Find the "newest" date
dates = links_mutations.str.split('\\/').apply(lambda x: x[-1])
dates = pd.to_datetime(dates, format='CFTR2_%d%B%Y.xlsx',errors='coerce')
file_link = links_mutations.loc[dates.idxmax()]
filename = file_link.split('/')[-1]

# Download the file
d = s.get(file_link)
path_excel = os.path.join(dir_data, filename)
fd = open(path_excel, 'wb')
fd.write(d.content)
fd.close()

# Get the JSON file-version of the mutations 
json_mutations = s.get('https://cftr2.org/json/cftr2_mutations').json()
# Encode to html
html_mutations = [quote(json) for json in json_mutations]
# Create lookup dictionaries
json2html = dict(zip(json_mutations, html_mutations))
html2json = dict(zip(html_mutations, json_mutations))


####################################
### --- (2) PROCESS VARIANTS --- ###

dat_variants = pd.read_excel(path_excel, skiprows=10)
di_rename = {"Variant cDNA name\n(ordered 5' to 3')":'cDNA_name', 'Variant protein name':'protein_name', 'Variant legacy name':'legacy_name', '# alleles in CFTR2':'num_alleles', 'Allele frequency in CFTR2\n(of 142,036 identified variants)*':'allele_freq', '% pancreatic insufficient (patients with variant in trans with ACMG-PI variant, with variant in homozygosity, or with another variant expected to lead to no CFTR protein production)':'pct_PI', "Variant final determination\n29 April 2022 (current version)":'cf_causing'}
dat_variants = dat_variants[list(di_rename)].rename(columns=di_rename)
dat_variants = dat_variants[dat_variants.notnull().mean(1) == 1]
dat_variants.to_csv(os.path.join(dir_data, 'mutation_history.csv'), index=False)

legacy_not_json = list(np.setdiff1d(dat_variants['legacy_name'].unique(), json_mutations))
json_not_legacy = list(np.setdiff1d(json_mutations, dat_variants['legacy_name'].unique()))
print('Legacy names not in JSON: {}'.format(legacy_not_json))
print('JSON names not in legacy: {}'.format(json_not_legacy))


######################################
### --- (3) INDIVUDAL VARIANTS --- ###

path_uni = os.path.join(dir_data, f'cftr2_uni.csv')
if scrape_variants:
    # Loop over each mutation and extract the data
    holder = []
    for i, html_mutation in enumerate(html_mutations):
        print('Downloading data for mutation {} of {}'.format(i+1, len(html_mutations)))
        url_i = f'https://cftr2.org/mutation/general/{html_mutation}'
        page_i = s.get(url_i)
        tables_i = merge_html_tables(page_i.text)
        tables_i.insert(0, 'mutation', html2json[html_mutation])
        holder.append(tables_i)
        # Pause for up to one second
        sleep(np.random.uniform(0,1))

    # Concatenate all of the tables
    cn_ord = ['mutation', 'category', 'age', 'n_pat', 'msr', 'value']
    df_uni = pd.concat(holder)[cn_ord].sort_values(cn_ord)
    # Add on the variant information
    df_uni = df_uni.merge(dat_variants, left_on='mutation', right_on='legacy_name', how='left')
    df_uni['num_alleles'] = df_uni['num_alleles'].astype(int)
    # Save to CSV
    df_uni.to_csv(path_uni, index=False)
else:
    assert os.path.exists(path_uni), 'If the scrape_variants is False, then the file must exist'
    df_uni = pd.read_csv(path_uni)

# Find mutations we have at least some data for
u_mutations = df_uni[(df_uni['msr']=='mutant') & (df_uni['value']!='insufficient data')]['mutation'].unique()
print(f'Number of unique mutations with available data: {len(u_mutations)} out of {len(dat_variants)}')


##################################
### --- (4) JOINT VARIANTS --- ###

# Get all combinations of mutations
mut_combs = list(combinations(u_mutations, 2))
n_combs = len(mut_combs)
print('Number of unique mutation combinations: {}'.format(n_combs))
# Estimate the frequency by getting the product of the alleles
df_mut_combs = pd.DataFrame(mut_combs).rename_axis('comb').melt(ignore_index=False,var_name='idx',value_name='mutation').reset_index().merge(dat_variants[['legacy_name', 'num_alleles']],left_on='mutation', right_on='legacy_name').drop(columns='legacy_name')
df_mut_combs['num_alleles'] = df_mut_combs['num_alleles'].astype(int)
df_mut_combs = df_mut_combs.pivot(['comb'],'idx',['num_alleles','mutation'])
df_mut_combs.columns = ['_'.join([str(c) for c in col]) for col in df_mut_combs.columns]
df_mut_combs = df_mut_combs.assign(allele_prod=lambda x: x['num_alleles_0'] * x['num_alleles_1']).drop(columns=['num_alleles_0', 'num_alleles_1'])
df_mut_combs = df_mut_combs.sort_values('allele_prod', ascending=False).reset_index(drop=True)
# Add on the homozygous combinations
dat_homo = pd.DataFrame({'mutation_0':u_mutations, 'mutation_1':u_mutations}).merge(dat_variants[['legacy_name', 'num_alleles']],left_on='mutation_0', right_on='legacy_name').drop(columns='legacy_name')
dat_homo = dat_homo.assign(allele_prod = lambda x: (x['num_alleles']**2).astype(int)).sort_values('num_alleles',ascending=False).drop(columns='num_alleles').reset_index(drop=True)
# Merge together
df_mut_combs = pd.concat([dat_homo, df_mut_combs],axis=0).drop_duplicates().reset_index(drop=True)

# Path to write the csv
path_comb = os.path.join(dir_data, f'cftr2_comb.csv')

if scrape_combinations:
    # Loop over each combination and extract the data
    stime, holder_comb = time(), []
    chunked = df_mut_combs.head(num_variants).reset_index(drop=True)
    n_chunked = len(chunked)
    for i, r in chunked.iterrows():
        mut_comb = (r['mutation_0'], r['mutation_1'])
        if (i+1) % 10 == 0:
            print('Combination {} of {}'.format(i+1, n_chunked))
            print(f'A total of {len(holder_comb)} have been found so far')
            dtime, n_left = time() - stime, n_chunked - (i+1)
            rate = (i+1) / dtime
            print(f'Estimated time remaining: {n_left / rate/ 60:.0f} minutes')
        # Get the URL
        mut_comb_html = [json2html[m] for m in mut_comb]
        url_i = f'https://cftr2.org/mutation/general/{"/".join(mut_comb_html)}'
        page_i = s.get(url_i)
        is_found = 'The consequences of this variant' not in page_i.text
        if is_found:
            tables_i = merge_html_tables(page_i.text)
            tables_i.insert(0, 'mutation2', mut_comb[1])
            tables_i.insert(0, 'mutation1', mut_comb[0])
            holder_comb.append(tables_i)
        if ((i+1) % 1000 == 0) or ((i+1) == n_chunked):  # Do an early save
            # Concatenate all of the tables
            cn_ord = ['mutation1', 'mutation2', 'category', 'age', 'n_pat', 'msr', 'value']
            df_comb = pd.concat(holder_comb)[cn_ord].sort_values(cn_ord).reset_index(drop=True)
            # Merge with existing
            df_comb = merge_frame_with_existing(df=df_comb, path=path_comb)
            # Save to CSV
            df_comb.to_csv(path_comb, index=False)
            holder_comb = []
else:
    df_comb = pd.read_csv(path_comb)

s.close()


##################################
### --- (5) CFTR1 DATABASE --- ###

# Define URLs
base_url = 'http://www.genet.sickkids.on.ca/'
url = base_url + 'SearchPage,$Form.direct'

# Payload from the Request URL: http://www.genet.sickkids.on.ca/SearchPage,$Form.direct
payload = {'formids':'PropertySelection,mutationSearchValue',
           'seedids':'ZH4sIAAAAAAAAAFvzloG1vI6hRqc4tagsMzlVxUCnIDEdRCXn5xbk56XmlYDZeSWJmXmpRUB2cWpxcWZ+HohVACSc8otSwOLBGak5OWCBlEogFQA0xSczLxvIdMxKrAguSSwpLQZyglLzgOqhitzyi3JVDADDTn+1hAAAAA==',
           'submitmode':'',
           'submitname':'',           
           'PropertySelection':'position',
           'mutationSearchValue':''}  # Will vary over the loop

if scrape_cftr1:
    print('Scraping CFTR1 database')
    # Since there are 1480 amino acids, we expect up to ~4500 position locations
    n_pos = 4500
    holder = []
    stime = time()
    for i in range(1, n_pos+1):
        payload['mutationSearchValue'] = str(i)
        if (i+1) % 5 == 0:
            print(f'Iteration {i+1} of {n_pos}\nNumber of results={len(holder)}')
            dtime, nleft = time() - stime, n_pos - (i+1)
            rate = (i+1) / dtime
            meta = (nleft / rate) / 60
            print(f'ETA = {meta:.1f} minutes remaining')
        s = requests.Session()
        r = s.post(url, data=payload)
        assert r.status_code == 200, 'Status code suggests failure'
        if 'Consequence' in r.text:
            soup = BeautifulSoup(r.text)
            # Should be the last table in the list
            soup_tbl = soup.find_all('table')[-1]
            # Each row should be associated with a different <tr>
            tbl_rows = soup_tbl.find_all('tr')
            # Get the column headers
            cols = tbl_rows[0].text.replace('\n',',').replace('\r','')
            cols = re.sub(r'^\,|\,$','',re.sub(r'\,{1,}',',',cols)).split(',')
            assert len(cols) == 6, 'Expected 6 columns'
            # Process the rows
            rows = [r.find_all('td') for r in tbl_rows[1:]]
            assert all([len(r)==6 for r in rows]), 'Expected 6 columns per row'
            # Clean each row up
            rows = [pd.Series([c.text for c in r]).str.replace('\\n|\\r|\\t','',regex=True).str.replace('\\s{2,}','',regex=True).str.strip().to_list() for r in rows]
            # Combine into a DF
            df = pd.DataFrame.from_records(rows,columns=cols)
            # Add on the URLs
            links = [base_url + a['href'] for a in soup_tbl.find_all('a', href=True)]
            cDNA_names = pd.Series([a.text.replace('\r','').replace('\t','').replace('\n','') for a in soup_tbl.find_all('a', href=True)]).str.strip()
            assert len(links) == len(df), 'links does not align with rows'
            assert (df['cDNA Name'] == cDNA_names).all(), 'cDNA name does not align'
            df.insert(df.shape[1], 'url', links)
            holder.append(df)
        s.close()
        # Pause for 0.5-1.5 seconds
        sleep(np.random.rand() + 0.5)
    # Merge and save results
    dat_cftr1 = pd.concat(holder).reset_index(drop=True)
    print(f'A total of {len(dat_cftr1)} mutations found')
    dat_cftr1.to_csv(os.path.join(dir_data, 'cftr1.csv'),index=False)


print('~~~ End of 1_scrape_cftr2.py ~~~')