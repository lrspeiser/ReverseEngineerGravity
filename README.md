# Reverse Engineer Gravity - Gaia Data Analysis

This project analyzes Gaia satellite data to understand the distribution of stars across different regions of the Milky Way and reverse engineer gravitational physics.

## 📁 Project Structure

```
ReverseEngineerGravity/
├── README.md                           # This file
├── reverse_engineer_gravity.py         # Main gravity reverse engineering script
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
│   └── gaia_analysis_detailed_plots.png # Detailed analysis plots
└── reports/                           # Analysis reports
    └── gaia_insights_report.txt       # Comprehensive insights report
```

## 🚀 Quick Start

### 1. Set up the environment
```bash
conda activate gravity_reverse
```

### 2. Create Gaia data summary
```bash
cd scripts
python create_gaia_summary.py
```

### 3. Analyze the summary data
```bash
cd analysis
python analyze_gaia_summary.py
```

### 4. Run gravity reverse engineering
```bash
python reverse_engineer_gravity.py
```

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
- **Modified gravity theory testing**
- **Stellar population analysis**

## 📈 Generated Files

### Data Files:
- `data/gaia_summary/gaia_distance_summary.csv`: Distance bin statistics
- `data/gaia_processed/gaia_processed_data.csv`: Full processed dataset

### Visualizations:
- `plots/gaia_summary_plots.png`: Basic summary plots
- `plots/gaia_analysis_detailed_plots.png`: Detailed analysis plots

### Reports:
- `reports/gaia_insights_report.txt`: Comprehensive analysis report

## 🛠️ Scripts

### `scripts/create_gaia_summary.py`
- Processes raw Gaia data
- Calculates galactocentric coordinates
- Creates distance-based summaries
- Generates basic visualizations

### `analysis/analyze_gaia_summary.py`
- Analyzes summary statistics
- Generates insights by galactic region
- Creates detailed visualizations
- Produces comprehensive reports

### `reverse_engineer_gravity.py`
- Main gravity reverse engineering script
- Uses processed Gaia data for physics analysis
- Implements neural network-based gravity modeling

## 📋 Requirements

- Python 3.10
- PyTorch
- NumPy, Pandas, Matplotlib
- SciPy, Scikit-learn
- Astropy

## 🔄 Workflow

1. **Data Processing**: Raw Gaia data → Processed coordinates → Distance summaries
2. **Analysis**: Summary statistics → Regional insights → Visualizations
3. **Physics**: Processed data → Neural network training → Gravity reverse engineering

## 📝 Notes

- The solar neighborhood (7-9 kpc) is the best-sampled region
- Star density peaks at 9.0 kpc from the galactic center
- The rotation curve shows significant variation with radius
- Stars are concentrated in a thin disk structure 