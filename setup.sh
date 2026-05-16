#!/bin/bash

# create environment
ENV="05_image_restoration"
conda create -y -n "$ENV" python=3.10
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
    pip install git+https://github.com/dlmbl/dlmbl-unet
    python -m ipykernel install --user --name "05_image_restoration"
    # Clone the extra repositories
    git clone https://github.com/krulllab/COSDD.git 04_bonus_COSDD/COSDD
    pip install -U tensorboard
fi

# Download the data
# CARE + N2V
if [ ! -d "data/denoising-N2V_SEM.unzip" ] || [ ! -d "data/denoising-CARE_U2OS.unzip" ]; then
    echo "Downloading CARE + N2V data..."
    python download_careamics_portfolio.py
else
    echo "CARE, N2V data already exists, skipping download."
fi

cd 04_bonus_COSDD/
# COSDD
if [ ! -d "checkpoints" ]; then
    echo "Adding pretrained checkpoint..."
    mkdir checkpoints
fi
cd checkpoints/
if [ ! -d "mito-pretrained" ]; then
    cp -r /mnt/efs/aimbl_2025/data/mito-pretrained .
fi
cd ../../
