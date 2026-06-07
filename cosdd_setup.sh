#!/bin/bash

echo "======================================"
echo "Creating environment for COSDD..."
echo "======================================"
ENV="05_image_restoration_COSDD"
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
    pip install torch torchvision
    pip install lightning ipykernel matplotlib tifffile scikit-learn scikit-image tensorboard ipywidgets

    # Clone the COSDD repository
    git clone https://github.com/krulllab/COSDD.git 04_bonus_COSDD/COSDD
    cd 04_bonus_COSDD/COSDD
    git checkout c0b49ac
fi

# other preparations
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
