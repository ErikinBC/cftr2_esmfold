#!/bin/bash

# Installation steps based on: https://github.com/facebookresearch/esm
# # after instance is up and running, do the following steps:
# i) get the ip address, and save it as a variable:
#       lambda_ip=ubuntu@XXX.XXX.XXX.XXX
#       path_cftr2=~/cftr2_esmfold
# ii) check that ssh works:
#       ssh -i ~/.ssh/id_rsa $lambda_ip
# iii) copy over the files: 
        # scp -i ~/.ssh/id_rsa $path_cftr2/lambda_cloud_setup.sh $lambda_ip:~/
        # scp -i ~/.ssh/id_rsa $path_cftr2/5_esm_fold.py $lambda_ip:~/
        # scp -i ~/.ssh/id_rsa $path_cftr2/data/cftr_polypeptides.csv $lambda_ip:~/
# iv) On VSCode, go to remote explorer, configure the .ssh/config file with the IP address (e.g. XXX.XXX.XXX.XX), and then connect to session
# v) Run the module:
#       python3 5_esm_fold.py --chunk_size 64 --num_recycles 3 --chain_linker 25 --fp_precision 16 --min_amino_acids 100
# vi) Zip the folder: 
#       tar -zcvf esm_fold.tar.gz data/esmfold
# vii) Download the data: 
#       scp -i ~/.ssh/id_macbook $lambda_ip:~/data/esmfold/*.npy $path_cftr2/data/esmfold
#       scp -i ~/.ssh/id_rsa $lambda_ip:~/esm_fold.tar.gz $path_cftr2
#       rsync -a --ignore-existing $path_cftr2/data/esmfold $lambda_ip:~/data/esmfold/*.npy

# install conda if it does not exist
path2conda=$(which conda)
npath2conda=${#path2conda}
if [ $npath2conda -eq 0 ]; then
    echo "conda is not installed, installing now"
    wget "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh"
    bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
    # Close the terminal
    exit
else
    echo "conda already installed"
    conda create -n esm python=3.9
    conda activate esm
fi

# Set up the background models and packages
echo "--- setting up aria2c ---"
sudo apt-get install aria2 -qq
aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/esmfold.model &

# pip install specific versions of packages
echo "--- setting up pip-required packaged ---"
pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol
pip install -q git+https://github.com/NVIDIA/dllogger.git

# do a conda install of pybind11 to prevent errors with openfold
echo "--- conda-required packages ---"
conda install pybind11 pandas

# install openfold
echo "--- installing openfold ---"
pip install -q git+https://github.com/aqlaboratory/openfold.git@6908936b68ae89f67755240e2f588c09ec31d4c8

# install esmfold
echo "--- installing esmfold ---"
pip install -q git+https://github.com/sokrypton/esm.git

# install other packages
echo "--- installing other packages ---"
pip install dm-tree
pip install scipy

# Set up the data directory to load in/save
mkdir data
mv cftr_polypeptides.csv data/


echo "~~~ End of lambda_cloud_setup.sh ~~~"