#!/bin/bash

echo "Installing JAX with CUDA support for RTX 5090..."

# Update pip
pip install --upgrade pip

# Install JAX with CUDA 12 support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install flax optax numpy pandas matplotlib scipy scikit-learn

echo "Installation complete!"
echo "To verify JAX is working with GPU, run:"
echo "python -c \"import jax; print('JAX devices:', jax.devices())\"" 