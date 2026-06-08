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

# create environment for MicroSplit
echo "======================================"
echo "Creating environment for MicroSplit..."
echo "======================================"
ENV="$HOME/.virtualenvs/05_image_restoration_microsplit"
uv venv --python 3.11 "$ENV"
source "$ENV/bin/activate"

uv pip install \
    "git+https://github.com/CAREamics/MicroSplit-reproducibility.git" \
    tensorboard \
    "setuptools<81" \
    ipykernel
python -m ipykernel install --user --name "05_image_restoration_microsplit" \
    --display-name "05 Image Restoration (MicroSplit)"

deactivate

# create environment for CARE & N2V exercises
echo "======================================================"
echo "Creating environment for CARE and Noise2Void..."
echo "======================================================"
ENV="$HOME/.virtualenvs/05_image_restoration"
uv venv --python 3.11 "$ENV"
source "$ENV/bin/activate"

uv pip install \
    careamics \
    careamics_portfolio \
    "git+https://github.com/dl-janelia/dlmbl-unet" \
    tensorboard \
    "setuptools<81" \
    ipykernel
python -m ipykernel install --user --name "05_image_restoration" \
    --display-name "05 Image Restoration (CARE/N2V)"

# Download the data (the 05_image_restoration env is still active)
# CARE + N2V
if [ ! -d "data/denoising-N2V_SEM.unzip" ] || [ ! -d "data/denoising-CARE_U2OS.unzip" ]; then
    echo "Downloading CARE + N2V data..."
    python download_careamics_portfolio.py
else
    echo "CARE, N2V data already exists, skipping download."
fi

deactivate
