#!/usr/bin/env python3
"""
plot_rotation_curves.py

Create visualization of Milky Way rotation curves using Gaia DR3 subset
and the gravity model learned from the 7,000-epoch training.

Outputs:
    plots/rotation_curves.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_PATH = Path('data/gaia_processed/gaia_processed_data.csv')
CHECKPOINT_PATH = Path('checkpoint_epoch_6000.json')  # latest saved params
PLOT_PATH = Path('plots/rotation_curves.png')

# -----------------------------------------------------------------------------
# Helper functions (mirrored from training script)
# -----------------------------------------------------------------------------
G = 4.302e-6  # kpc^3 / (M_sun * Myr^2)

def calculate_density(R_kpc: np.ndarray) -> np.ndarray:
    """Simple exponential disk density model in M_sun/kpc^3."""
    rho_0 = 0.1  # M_sun/pc^3 at solar radius
    R_d = 3.0    # kpc scale length
    rho = rho_0 * np.exp(-R_kpc / R_d) * 1e9  # convert pc^3 -> kpc^3
    return rho

def calculate_newtonian_velocity(R_kpc: np.ndarray, M_disk_thin: float = 8.302e10, R_d_thin: float = 2.963) -> np.ndarray:
    """Very simplified enclosed-mass model to estimate Newtonian velocity (km/s)."""
    M_enclosed = M_disk_thin * (1 - np.exp(-R_kpc / R_d_thin))
    v_newton = np.sqrt(G * M_enclosed / R_kpc) * 3.086e13 / 3.154e13  # convert kpc/Myr -> km/s
    return v_newton

def xi_function(rho: np.ndarray, R: np.ndarray, rho_c: float, n_exp: float, A_boost: float) -> np.ndarray:
    """Compute enhancement factor xi(ρ, R) from model parameters."""
    density_factor = 1 / (1 + (rho / rho_c) ** n_exp)
    radial_factor = 1 + A_boost * np.exp(-R / 10.0)
    xi = 1 + density_factor * radial_factor
    return xi

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("Loading Gaia processed data …")
df = pd.read_csv(DATA_PATH)
mask = (df['R_kpc'] > 6) & (df['R_kpc'] < 18) & (df['sigma_v'] < 50)
df = df[mask]
R = df['R_kpc'].values
v_obs = df['v_circ'].values

# -----------------------------------------------------------------------------
# Load model parameters from checkpoint
# -----------------------------------------------------------------------------
if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH) as f:
        ckpt = json.load(f)
    params = ckpt.get('model_params', {})
    rho_c = params.get('rho_c', 1e20)
    n_exp = params.get('n_exp', 1.0)
    A_boost = params.get('A_boost', 0.1)
else:
    print(f"Warning: checkpoint {CHECKPOINT_PATH} not found, using defaults")
    rho_c, n_exp, A_boost = 1e20, 1.0, 0.1

print(f"Using model parameters: rho_c={rho_c:.3e}, n={n_exp:.3f}, A={A_boost:.3f}")

# -----------------------------------------------------------------------------
# Compute curves
# -----------------------------------------------------------------------------
print("Computing Newtonian curve …")
v_newton = calculate_newtonian_velocity(R)
print("Computing modified gravity curve …")
rho_vals = calculate_density(R)
xi_vals = xi_function(rho_vals, R, rho_c, n_exp, A_boost)
v_model = np.sqrt(xi_vals) * v_newton

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
print("Plotting …")
plt.figure(figsize=(10, 6))
# Observed rotation curve (scatter)
plt.scatter(R, v_obs, s=5, alpha=0.3, label='Gaia DR3 (observed)')

# Newtonian prediction
# To get smooth curve, sort by R
idx_sort = np.argsort(R)
plt.plot(R[idx_sort], v_newton[idx_sort], color='black', lw=2, label='Newtonian (baryons only)')

# Modified gravity prediction
plt.plot(R[idx_sort], v_model[idx_sort], color='red', lw=2, label='Learned Model (7000 epochs)')

plt.xlabel('Galactocentric Radius R (kpc)')
plt.ylabel('Circular Velocity v_circ (km/s)')
plt.title('Milky Way Rotation Curve')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_PATH, dpi=300)
print(f"Saved plot to {PLOT_PATH}")
