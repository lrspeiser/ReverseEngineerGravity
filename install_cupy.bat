@echo off
echo Installing CuPy with CUDA support for RTX 5090...

REM Update pip
python -m pip install --upgrade pip

REM Install CuPy with CUDA 12 support
python -m pip install cupy-cuda12x

REM Install other dependencies
python -m pip install numpy pandas matplotlib scipy scikit-learn

echo Installation complete!
echo To verify CuPy is working with GPU, run:
echo python -c "import cupy as cp; print('CuPy version:', cp.__version__); print('CUDA available:', cp.cuda.is_available())"
pause 