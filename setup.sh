#!/bin/bash

# create environment
ENV="05_image_restoration"
conda create -y -n "$ENV" python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh" # init conda
conda activate "$ENV"

# check that the environment was activated
if [[ "$CONDA_DEFAULT_ENV" == "$ENV" ]]; then
    echo "Environment activated successfully"
else
    echo "Failed to activate the environment"
fi

# Further instructions that should only run if the environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENV" ]]; then
    pip install careamics
    pip install careamics_portfolio
    pip install git+https://github.com/dl-janelia/dlmbl-unet
    pip install tensorboard
    pip install "setuptools<81"  # setuptools>=81 removes pkg_resources, required by tensorboard<=2.20

    # packages to run jupyter notebooks
    python -m ipykernel install --user --name "05_image_restoration"

    # Clone the extra COSDD repository
    git clone https://github.com/krulllab/COSDD.git 04_bonus_COSDD/COSDD
fi

# Download the data
# CARE + N2V
if [ ! -d "data/denoising-N2V_SEM.unzip" ] || [ ! -d "data/denoising-CARE_U2OS.unzip" ]; then
    echo "Downloading CARE + N2V data..."
    python download_careamics_portfolio.py
else
    echo "CARE, N2V data already exists, skipping download."
fi

# COSDD
cd 04_bonus_COSDD/
if [ ! -d "checkpoints" ]; then
    echo "Adding pretrained checkpoint..."
    mkdir checkpoints
fi
cd checkpoints/
if [ ! -d "mito-pretrained" ]; then
    cp -r /mnt/efs/dl_jrc/data/05_image_restoration/COSDD/mito-pretrained .
fi
cd ../../
