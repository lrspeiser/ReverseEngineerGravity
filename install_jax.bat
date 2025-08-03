@echo off
echo Installing JAX with CUDA support for RTX 5090...

REM Update pip
python -m pip install --upgrade pip

REM Install JAX with CUDA 12 support
python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

REM Install other dependencies
python -m pip install flax optax numpy pandas matplotlib scipy scikit-learn

echo Installation complete!
echo To verify JAX is working with GPU, run:
echo python -c "import jax; print('JAX devices:', jax.devices())"
pause 