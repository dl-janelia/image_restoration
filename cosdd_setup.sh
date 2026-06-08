#!/bin/bash

# ----------------------------------------------------------------------
# Ensure uv is installed
# ----------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    # the installer drops uv in ~/.local/bin; make it available in this session
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed: $(uv --version)"
fi

# create environment for COSDD
echo "======================================"
echo "Creating environment for COSDD..."
echo "======================================"
ENV="$HOME/.virtualenvs/05_image_restoration_COSDD"
uv venv --python 3.11 "$ENV"
source "$ENV/bin/activate"

uv pip install torch torchvision
uv pip install lightning ipykernel matplotlib tifffile scikit-learn scikit-image tensorboard ipywidgets
python -m ipykernel install --user --name "05_image_restoration_COSDD" \
    --display-name "05 Image Restoration (COSDD)"

# Clone the COSDD repository
if [ ! -d "04_bonus_COSDD/COSDD" ]; then
    git clone https://github.com/krulllab/COSDD.git 04_bonus_COSDD/COSDD
    git -C 04_bonus_COSDD/COSDD checkout eae0b6c
else
    echo "COSDD repository already exists, skipping clone."
fi

deactivate

# other preparations
if [ ! -d "04_bonus_COSDD/checkpoints" ]; then
    echo "Adding pretrained checkpoint..."
    mkdir 04_bonus_COSDD/checkpoints
fi
if [ ! -d "04_bonus_COSDD/checkpoints/mito-pretrained" ]; then
    cp -r /mnt/efs/dl_jrc/data/05_image_restoration/COSDD/mito-pretrained 04_bonus_COSDD/checkpoints/
fi
