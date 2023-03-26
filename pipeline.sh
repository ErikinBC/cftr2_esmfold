#!/bin/bash

# Script order to replicate analysis pipeline
conda activate cftr

echo "--- (1) Scrape CFTR2 variants ---"
python3 1_scrape_cftr2.py --scrape_variants --scrape_combinations --num_variants 25000

echo "--- (2) Get the NCBI links ---"
python3 2_get_ncbi.py

echo "--- (3) Scrape the NCBI data ---"
python3 3_scrape_ncbi.py

echo "--- (4) Get the CFTR gene ---"
python3 4_cftr_gene.py --ensembl_version 109

echo "--- (5) Run ESMFold ---"
python3 5_esm_fold.py --chunk_size 64 --num_recycles 3 --chain_linker 25 --fp_precision 16 --min_amino_acids 100

echo "--- (6) Generate the X/y data ---"
python3 6_process_Xy.py

echo "--- (7) Generate summary stats and figures ---"
python3 7_summary_stats.py

echo "--- (8) Adjust for AminoAcid Length with NW-estimator ---"
python3 8_debiay_y.py

echo "--- (9) Run ML model to predict adjusted labels ---"
python3 9_predict_y.py

echo "~~~ End of pipeline.sh ~~~"