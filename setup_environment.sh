#!/bin/bash
# Setup script for Integrative Network Modeling (INM) - Bayesian Network Inference
# This script creates a conda environment and installs all required packages

set -e  # Exit on error

ENV_NAME="inm-py310"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "INM Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '${ENV_NAME}' already exists!"
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Activating it..."
        echo "To activate: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create conda environment
echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install conda packages
echo ""
echo "Installing conda packages..."
conda install -n ${ENV_NAME} -c conda-forge numpy scipy -y

# Install pip packages
echo ""
echo "Installing pip packages..."
pip install --upgrade pip

# Core packages
pip install numpyro>=0.18.0
pip install arviz>=0.23.0

# JAX - Check for GPU
echo ""
read -p "Do you have a CUDA-capable GPU? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing JAX with CUDA support..."
    read -p "CUDA version? (11 or 12, default: 12): " cuda_version
    cuda_version=${cuda_version:-12}
    if [ "$cuda_version" == "12" ]; then
        pip install jax[cuda12] jax-cuda12-plugin
    elif [ "$cuda_version" == "11" ]; then
        pip install jax[cuda11] jax-cuda11-plugin
    else
        echo "Installing CPU-only JAX..."
        pip install jax jaxlib
    fi
else
    echo "Installing CPU-only JAX..."
    pip install jax jaxlib
fi

# Data processing
pip install pandas>=2.0.0

# Visualization
pip install matplotlib>=3.8.0 seaborn>=0.13.0

# Network analysis
pip install networkx>=3.0 python-louvain>=0.16

# Excel I/O
pip install openpyxl>=3.1.0

# Configuration
pip install pyyaml>=6.0

# CLI tools
pip install click>=8.0

# Optional packages
pip install scikit-learn>=1.3.0

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import numpyro; print('✓ NumPyro:', numpyro.__version__)"
python -c "import jax; print('✓ JAX:', jax.__version__)"
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import pandas; print('✓ Pandas:', pandas.__version__)"
python -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)"
python -c "import networkx; print('✓ NetworkX:', networkx.__version__)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify JAX can see your GPU (if applicable), run:"
echo "  python -c 'import jax; print(jax.devices())'"
echo ""
