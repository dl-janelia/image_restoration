#!/bin/bash

# create environment
ENV="05_image_restoration"
conda create -y -n "$ENV" python=3.11
conda activate "$ENV"

# check that the environment was activated
if [[ "$CONDA_DEFAULT_ENV" == "$ENV" ]]; then
    echo "Environment activated successfully"
else
    echo "Failed to activate the environment"
fi

# Further instructions that should only run if the environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENV" ]]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install git+https://github.com/CAREamics/MicroSplit-reproducibility.git
    pip install tensorboard torch_tb_profiler scikit-learn gdown jupyterlab
    # Using pytorch-lightning 2.4.0 causes bugs in tensorboard and interupting training.
    pip install pytorch-lightning==2.3.3
    pip install git+https://github.com/dlmbl/dlmbl-unet # TODO: potentially replace with `git+https://github.com/dl-janelia/dlmbl-unet` 
    python -m ipykernel install --user --name "05_image_restoration"
    # Clone the extra repositories
    git clone https://github.com/krulllab/COSDD.git 03_COSDD/COSDD
    pip install -U tensorboard
    pip install "setuptools<=81"  # setuptools>=82 removes pkg_resources, required by tensorboard<=2.20
    pip install careamics_portfolio
    pip install tifffile matplotlib
fi

# Download the data
# CARE + N2V
if [ ! -d "data/denoising-N2V_SEM.unzip" ] || [ ! -d "data/denoising-CARE_U2OS.unzip" ] || [ ! -d "data/denoising-N2N_SEM.unzip" ]; then
    echo "Downloading CARE + N2V data..."
    python download_careamics_portfolio.py
else
    echo "CARE, N2V + N2N data already exists, skipping download."
fi

# COSDD
cd 03_COSDD/
if [ ! -d "checkpoints" ]; then
    echo "Adding pretrained checkpoint..."
    mkdir checkpoints
fi
cd checkpoints/
if [ ! -d "mito-pretrained" ]; then
    cp -r /mnt/efs/aimbl_2025/data/mito-pretrained . # FIXME: this is not there...
fi
cd ../../
