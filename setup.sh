#!/bin/bash

# create environment for MicroSplit
echo "======================================"
echo "Creating environment for MicroSplit..."
echo "======================================"
ENV="05_image_restoration_microsplit"
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
    pip install git+https://github.com/CAREamics/MicroSplit-reproducibility.git
    
    # packages to run jupyter notebooks
    pip install tensorboard
    pip install "setuptools<81"  # setuptools>=81 removes pkg_resources, required by tensorboard<=2.20
    pip install ipykernel
    python -m ipykernel install --user --name "05_image_restoration"
fi


# create environment for CARE & N2V exercises
echo "======================================================"
echo "Creating environment for CARE, Noise2Void and COSDD..."
echo "======================================================"
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
    pip install ipykernel
    python -m ipykernel install --user --name "05_image_restoration"
fi

# Download the data
# CARE + N2V
if [ ! -d "data/denoising-N2V_SEM.unzip" ] || [ ! -d "data/denoising-CARE_U2OS.unzip" ]; then
    echo "Downloading CARE + N2V data..."
    python download_careamics_portfolio.py
else
    echo "CARE, N2V data already exists, skipping download."
fi
