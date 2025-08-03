# Reverse Engineer Gravity - Gaia Data Analysis (JAX GPU)

This project analyzes Gaia satellite data to understand the distribution of stars across different regions of the Milky Way and reverse engineer gravitational physics using JAX GPU-accelerated machine learning.

## 📁 Project Structure

```
ReverseEngineerGravity/
├── README.md                           # This file
├── reverse_engineer_gravity.py         # Main gravity reverse engineering script (JAX GPU)
├── requirements.txt                    # JAX dependencies with CUDA support
├── install_jax.sh                      # Linux/Mac JAX installation script
├── install_jax.bat                     # Windows JAX installation script
├── test_jax_setup.py                   # GPU verification script
├── gaia_sky_slices/                    # Raw Gaia data files
│   ├── all_sky_gaia.csv               # Main Gaia dataset
│   └── processed_*.parquet            # Processed sky slices
├── data/                               # Processed data files
│   ├── gaia_summary/                  # Summary statistics by distance
│   │   └── gaia_distance_summary.csv  # Distance bin statistics
│   └── gaia_processed/                # Full processed datasets
│       └── gaia_processed_data.csv    # Complete processed Gaia data
├── scripts/                           # Data processing scripts
│   └── create_gaia_summary.py         # Creates distance-based summaries
├── analysis/                          # Analysis and visualization scripts
│   └── analyze_gaia_summary.py        # Analyzes summary data and generates insights
├── plots/                             # Generated visualizations
│   ├── gaia_summary_plots.png         # Basic summary plots
│   ├── gaia_analysis_detailed_plots.png # Detailed analysis plots
│   └── reverse_engineered_gravity.png # Gravity model visualizations
└── reports/                           # Analysis reports
    └── gaia_insights_report.txt       # Comprehensive insights report
```

## 🚀 Quick Start

### 1. Install JAX with CUDA support (for RTX 5090)
```bash
# On Windows:
install_jax.bat

# On Linux/Mac:
chmod +x install_jax.sh
./install_jax.sh

# Or manually:
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax numpy pandas matplotlib scipy scikit-learn
```

### 2. Verify GPU setup
```bash
python -c "import jax; print('JAX devices:', jax.devices())"
```

### 3. Create Gaia data summary
```bash
cd scripts
python create_gaia_summary.py
```

### 4. Analyze the summary data
```bash
cd analysis
python analyze_gaia_summary.py
```

### 5. Run gravity reverse engineering (5000 epochs on GPU)
```bash
python reverse_engineer_gravity.py
```

## 🎯 **What This Does**

The main script performs **physics-informed neural network training** to reverse engineer gravity from Gaia rotation curve data:

- **5000 epochs** of GPU-accelerated training
- **Physics constraints**: Cassini spacecraft precision tests
- **Galactic validation**: Ensures model works across Milky Way scales
- **Formula extraction**: Converts neural network to analytical formulas
- **Multiple theories**: Tests MOND, Yukawa, Chameleon, and other gravity models

## 📊 Data Summary

The Gaia data has been processed and summarized by distance from the galactic center:

- **Total stars analyzed**: 143,992
- **Distance range**: 5.0 - 13.0 kpc from galactic center
- **Peak star density**: 1,508 stars/kpc³ at 9.0 kpc (solar neighborhood)
- **Rotation curve**: Variable (26-55 km/s)

### Key Findings by Region:

| Distance (kpc) | Stars | Density (stars/kpc³) | Mean Velocity (km/s) | Region Type |
|----------------|-------|---------------------|---------------------|-------------|
| 5.0 | 763 | 24.3 | 54.8 | Inner Galaxy |
| 7.0 | 55,270 | 1,256.6 | 37.1 | Solar Neighborhood |
| 9.0 | 85,271 | 1,507.9 | 30.3 | Solar Neighborhood |
| 11.0 | 2,549 | 36.9 | 32.5 | Outer Disk |
| 13.0 | 139 | 1.7 | 26.3 | Outer Disk |

## 🔬 Scientific Applications

This data is suitable for:
- **Rotation curve analysis** and dark matter constraints
- **Galactic structure studies**
- **Modified gravity theory testing** (MOND, Yukawa, Chameleon, etc.)
- **Stellar population analysis**
- **Physics-informed machine learning** for gravitational physics
- **Neural network to analytical formula conversion**

## 📈 Generated Files

### Data Files:
- `data/gaia_summary/gaia_distance_summary.csv`: Distance bin statistics
- `data/gaia_processed/gaia_processed_data.csv`: Full processed dataset
- `data/reverse_engineered_gravity_model.pkl`: Trained JAX model
- `data/gravity_formulas.json`: Extracted analytical formulas

### Visualizations:
- `plots/gaia_summary_plots.png`: Basic summary plots
- `plots/gaia_analysis_detailed_plots.png`: Detailed analysis plots
- `plots/reverse_engineered_gravity.png`: Gravity model results

### Reports:
- `reports/gaia_insights_report.txt`: Comprehensive analysis report

## 🛠️ Scripts

### `scripts/create_gaia_summary.py`
- Processes raw Gaia data
- Calculates galactocentric coordinates
- Creates distance-based summaries
- Generates basic visualizations

### `scripts/test_reverse_engineering.py`
- Tests the reverse engineering code
- Verifies data loading and processing
- Runs short training to check functionality

### `scripts/performance_test.py`
- Estimates training time on different devices
- Shows sample results from short training runs
- Creates quick visualizations for validation

### `scripts/device_comparison.py`
- Compares M1 Mac vs expected 5090 performance
- Benchmarks tensor operations
- Provides training time estimates

### `analysis/analyze_gaia_summary.py`
- Analyzes summary statistics
- Generates insights by galactic region
- Creates detailed visualizations
- Produces comprehensive reports

### `reverse_engineer_gravity.py` (Main Script)
- **JAX GPU-accelerated** physics-informed neural network
- **5000 epochs** of training on RTX 5090 GPU
- **Physics constraints**: Cassini spacecraft precision tests
- **Galactic validation**: Ensures model works across Milky Way scales
- **Formula extraction**: Converts neural network to analytical formulas
- **Multiple theories**: Tests MOND, Yukawa, Chameleon, and other gravity models
- **Real-time validation**: Checks model physics during training

## 📋 Requirements

### Core Dependencies
- **Python 3.10+**
- **JAX with CUDA 12 support** (optimized for RTX 5090)
- **Flax** (neural network library for JAX)
- **Optax** (optimization library for JAX)

### Data Science Stack
- **NumPy, Pandas, Matplotlib**
- **SciPy, Scikit-learn**
- **Astropy** (astronomical calculations)

### GPU Requirements
- **NVIDIA GPU** with CUDA 12 support
- **RTX 5090 recommended** for optimal performance
- **16GB+ VRAM** for large batch training

## 🔄 Workflow

1. **Setup**: Install JAX with CUDA support → Verify GPU setup → Test installation
2. **Data Processing**: Raw Gaia data → Processed coordinates → Distance summaries
3. **Analysis**: Summary statistics → Regional insights → Visualizations
4. **Physics**: Processed data → JAX neural network training → Gravity reverse engineering
5. **Validation**: Physics constraints → Galactic validation → Formula extraction

## 📝 Notes

### Data Characteristics
- The solar neighborhood (7-9 kpc) is the best-sampled region
- Star density peaks at 9.0 kpc from the galactic center
- The rotation curve shows significant variation with radius
- Stars are concentrated in a thin disk structure

### Performance Notes
- **JAX JIT compilation** provides significant speedup on GPU
- **5000 epochs** typically sufficient for convergence
- **Batch size 1024** optimized for RTX 5090 memory
- **Physics validation** ensures model learns correct behavior
- **Formula extraction** converts neural network to interpretable equations

### GPU Optimization
- **CUDA 12** provides best performance for RTX 5090
- **Memory management** handled automatically by JAX
- **Compiled training** reduces Python overhead
- **Functional programming** enables better optimization

## 🏗️ **JAX Architecture**

### Neural Network Design
- **Flax Module**: Physics-informed neural network with learnable parameters
- **Input**: [log(ρ), R/R_sun, z/kpc] - density, radius, height
- **Output**: ξ enhancement factor for gravity
- **Architecture**: 3 hidden layers [128, 64, 32] with ReLU activation
- **Physics Parameters**: ρ_c (critical density), n (exponent), A (amplitude)

### Training System
- **Optimizer**: AdamW with weight decay and learning rate scheduling
- **Loss Function**: MSE + Cassini constraint + physical regularization
- **JIT Compilation**: All training steps compiled for maximum speed
- **Batch Processing**: Efficient GPU memory usage with dynamic batching

### Physics Constraints
- **Cassini Test**: Ensures ξ ≈ 1 in solar system (precision: 2.3×10⁻⁵)
- **Galactic Gradient**: ξ increases from inner to outer galaxy
- **Density Screening**: High density regions suppress modifications
- **Edge Enhancement**: Outer galaxy shows significant enhancement

### Formula Extraction
- **Neural Network → Analytical**: Converts trained model to mathematical formulas
- **Multiple Theories**: Tests MOND, Yukawa, Chameleon, and other models
- **Validation**: Compares formulas against neural network predictions
- **Scoring**: Ranks formulas by accuracy and physical consistency

## 🔧 **Troubleshooting**

### JAX Installation Issues
```bash
# If JAX installation fails, try:
pip install --upgrade pip setuptools wheel
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Verify JAX sees GPU
python -c "import jax; print(jax.devices())"

# Run GPU test
python test_jax_setup.py
```

### Memory Issues
- **Reduce batch size** in `reverse_engineer_gravity.py` (line ~400)
- **Monitor GPU memory** with `nvidia-smi`
- **Close other GPU applications** during training

### Training Issues
- **Model not converging**: Increase epochs or adjust learning rate
- **Physics validation fails**: Check data quality and model architecture
- **Formula extraction fails**: Ensure model has learned non-trivial solution 