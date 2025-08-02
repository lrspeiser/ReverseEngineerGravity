#!/usr/bin/env python3
"""
reverse_engineer_gravity.py

Reverse engineer the formula for gravity from Gaia rotation curve data
using GPU-accelerated machine learning.

Requirements:
- PyTorch with CUDA support
- numpy, pandas, matplotlib
- scikit-learn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import time
from pathlib import Path

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class GravityReverseEngineer:
    """Main class for reverse engineering gravity from rotation curves."""
    
    def __init__(self, gaia_data_path='gaia_sky_slices/all_sky_gaia.csv'):
        self.gaia_data_path = gaia_data_path
        self.device = device
        
        # Physical constants
        self.G = 4.302e-6  # kpc³/M_sun/Myr²
        self.R_sun = 8.0   # kpc
        self.v_sun = 220.0 # km/s
        
        # Baryon model parameters (from your fits)
        self.baryon_params = {
            'M_disk_thin': 8.302e10,
            'R_d_thin': 2.963,
            'h_z_thin': 0.322,
            'M_disk_thick': 6.847e9,
            'R_d_thick': 5.836,
            'h_z_thick': 0.885,
            'M_bulge': 1.291e10,
            'a_bulge': 1.116,
            'M_gas': 1.626e10,
            'R_d_gas': 9.838,
            'h_z_gas': 0.399
        }
        
    def load_gaia_data(self):
        """Load and preprocess Gaia data."""
        print("Loading Gaia data...")
        df = pd.read_csv(self.gaia_data_path)
        
        # Filter to good quality data in 6-18 kpc range
        mask = (df['R_kpc'] > 6) & (df['R_kpc'] < 18) & (df['sigma_v'] < 50)
        df_filtered = df[mask].copy()
        
        print(f"Loaded {len(df_filtered)} stars in 6-18 kpc range")
        
        # Extract arrays
        self.R_data = df_filtered['R_kpc'].values
        self.v_data = df_filtered['v_obs'].values
        self.sigma_v = df_filtered['sigma_v'].values
        
        # Calculate density at each point
        self.rho_data = self.calculate_density(self.R_data)
        
        return df_filtered
    
    def calculate_density(self, R):
        """Calculate baryon density at given radii."""
        # Simplified - you'd use your full density model here
        rho_0 = 0.1  # M_sun/pc³ at solar radius
        R_d = 3.0    # kpc
        h_z = 0.3    # kpc
        
        rho = rho_0 * np.exp(-R/R_d) * 1e9  # Convert to M_sun/kpc³
        return rho
    
    def calculate_newtonian_velocity(self, R):
        """Calculate expected Newtonian velocity from baryons."""
        # Simplified - use your full model
        M_enclosed = self.baryon_params['M_disk_thin'] * (1 - np.exp(-R/self.baryon_params['R_d_thin']))
        v_newton = np.sqrt(self.G * M_enclosed / R) * 3.086e13 / 3.154e13  # Convert to km/s
        return v_newton
    
    def derive_empirical_xi(self):
        """Calculate required xi enhancement at each data point."""
        print("\nDeriving empirical xi values...")
        
        # Calculate Newtonian predictions
        v_newton = self.calculate_newtonian_velocity(self.R_data)
        
        # Required enhancement factor
        self.xi_empirical = (self.v_data / v_newton)**2
        
        # Clean up any infinities or extreme values
        mask = np.isfinite(self.xi_empirical) & (self.xi_empirical < 10)
        self.R_clean = self.R_data[mask]
        self.xi_clean = self.xi_empirical[mask]
        self.rho_clean = self.rho_data[mask]
        
        print(f"Xi enhancement ranges from {np.min(self.xi_clean):.2f} to {np.max(self.xi_clean):.2f}")
        
        # Bin and average for smoother curve
        R_bins = np.linspace(6, 18, 25)
        xi_binned = []
        R_centers = []
        
        for i in range(len(R_bins)-1):
            mask = (self.R_clean >= R_bins[i]) & (self.R_clean < R_bins[i+1])
            if np.sum(mask) > 10:
                xi_binned.append(np.median(self.xi_clean[mask]))
                R_centers.append((R_bins[i] + R_bins[i+1])/2)
        
        self.R_binned = np.array(R_centers)
        self.xi_binned = np.array(xi_binned)
        
        return self.R_binned, self.xi_binned

class PhysicsInformedNN(nn.Module):
    """Neural network that learns xi(ρ, R) with physical constraints."""
    
    def __init__(self, hidden_layers=[64, 64, 32]):
        super().__init__()
        
        # Input: [log(ρ), R/R_sun, z/kpc]
        layers = []
        input_size = 3
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_layers[i-1], hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable parameters for analytical constraints
        self.rho_c = nn.Parameter(torch.tensor(12.0))  # log10(rho_c)
        self.n_exp = nn.Parameter(torch.tensor(1.5))
        self.A_boost = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, rho, R, z=None):
        """
        Forward pass computing xi.
        
        Parameters:
        - rho: density in M_sun/kpc³
        - R: galactocentric radius in kpc
        - z: height above plane in kpc
        """
        if z is None:
            z = torch.zeros_like(R)
        
        # Prepare inputs (normalize and log-transform)
        log_rho = torch.log10(rho + 1e-10)
        R_norm = R / 8.0  # Normalize by R_sun
        z_norm = z / 0.5  # Normalize by scale height
        
        inputs = torch.stack([log_rho, R_norm, z_norm], dim=-1)
        
        # Neural network output
        nn_output = self.network(inputs).squeeze()
        
        # Physics-based modulation
        rho_ratio = rho / (10**self.rho_c)
        density_factor = 1 / (1 + rho_ratio**self.n_exp)
        
        # Combine NN with physics
        xi = 1 + self.A_boost * torch.sigmoid(nn_output) * density_factor
        
        return xi
    
    def cassini_constraint(self):
        """Calculate Cassini constraint violation."""
        rho_saturn = torch.tensor([2.3e21], device=self.network[0].weight.device)
        R_saturn = torch.tensor([9.5], device=self.network[0].weight.device)  # AU to kpc
        
        xi_saturn = self.forward(rho_saturn, R_saturn)
        cassini_violation = (xi_saturn - 1.0)**2 / (2.3e-5)**2
        
        return cassini_violation

class GravityTrainer:
    """Train the physics-informed neural network."""
    
    def __init__(self, engineer, model):
        self.engineer = engineer
        self.model = model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
    def prepare_data(self):
        """Prepare training data."""
        # Get empirical xi values
        R_binned, xi_binned = self.engineer.derive_empirical_xi()
        
        # Prepare full dataset
        R_tensor = torch.tensor(self.engineer.R_clean, dtype=torch.float32)
        rho_tensor = torch.tensor(self.engineer.rho_clean, dtype=torch.float32)
        xi_tensor = torch.tensor(self.engineer.xi_clean, dtype=torch.float32)
        
        # Create dataset
        dataset = TensorDataset(rho_tensor, R_tensor, xi_tensor)
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        return train_dataset, val_dataset
    
    def train(self, epochs=1000, cassini_weight=100.0):
        """Train the model."""
        print(f"\nTraining on {device}...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for rho, R, xi_target in self.train_loader:
                rho, R, xi_target = rho.to(device), R.to(device), xi_target.to(device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                xi_pred = self.model(rho, R)
                
                # MSE loss
                mse_loss = nn.MSELoss()(xi_pred, xi_target)
                
                # Cassini constraint
                cassini_loss = self.model.cassini_constraint()
                
                # Physical regularization
                reg_loss = 0.01 * (torch.abs(self.model.n_exp - 1.5) + 
                                   torch.abs(self.model.A_boost - 2.0))
                
                # Total loss
                loss = mse_loss + cassini_weight * cassini_loss + reg_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for rho, R, xi_target in self.val_loader:
                    rho, R, xi_target = rho.to(device), R.to(device), xi_target.to(device)
                    xi_pred = self.model(rho, R)
                    val_loss += nn.MSELoss()(xi_pred, xi_target).item()
            
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                print(f"  Parameters: rho_c = 10^{self.model.rho_c.item():.2f}, "
                      f"n = {self.model.n_exp.item():.2f}, A = {self.model.A_boost.item():.2f}")
                
                # Check Cassini
                with torch.no_grad():
                    cassini = self.model.cassini_constraint().item()
                    print(f"  Cassini violation: {cassini:.2e}")
        
        return train_losses, val_losses
    
    def extract_formula(self):
        """Extract analytical formula from trained model."""
        print("\nExtracting analytical formula...")
        
        # Test points
        R_test = np.linspace(1, 30, 100)
        rho_test = np.logspace(3, 21, 100)
        
        # Create mesh
        R_mesh, rho_mesh = np.meshgrid(R_test, rho_test)
        
        # Evaluate model
        with torch.no_grad():
            R_tensor = torch.tensor(R_mesh.flatten(), dtype=torch.float32).to(device)
            rho_tensor = torch.tensor(rho_mesh.flatten(), dtype=torch.float32).to(device)
            xi_pred = self.model(rho_tensor, R_tensor).cpu().numpy()
        
        xi_mesh = xi_pred.reshape(R_mesh.shape)
        
        # Fit simple analytical forms
        print("\nTrying analytical fits...")
        
        # 1. Extended power law
        def extended_power_law(x, a, b, c, d, e):
            R, log_rho = x
            return 1 + a * (R/8)**b / (1 + 10**(c * (log_rho - d)))**e
        
        # 2. Double exponential
        def double_exp(x, a, b, c, d):
            R, log_rho = x
            return 1 + a * (1 - np.exp(-R/b)) * np.exp(-10**(log_rho - c) / 10**d)
        
        # Prepare data for fitting
        R_fit = R_mesh.flatten()
        log_rho_fit = np.log10(rho_mesh.flatten())
        xi_fit = xi_mesh.flatten()
        
        # Remove any NaN or inf
        mask = np.isfinite(xi_fit)
        R_fit = R_fit[mask]
        log_rho_fit = log_rho_fit[mask]
        xi_fit = xi_fit[mask]
        
        # Try fits
        try:
            popt1, _ = curve_fit(extended_power_law, (R_fit, log_rho_fit), xi_fit, 
                                p0=[2, 0.5, 1, 12, 2], maxfev=5000)
            print(f"\nExtended power law: ξ = 1 + {popt1[0]:.3f}*(R/8)^{popt1[1]:.3f} / "
                  f"(1 + 10^({popt1[2]:.3f}*(log(ρ) - {popt1[3]:.3f})))^{popt1[4]:.3f}")
        except:
            print("Extended power law fit failed")
        
        try:
            popt2, _ = curve_fit(double_exp, (R_fit, log_rho_fit), xi_fit,
                                p0=[2, 10, 12, 3], maxfev=5000)
            print(f"\nDouble exponential: ξ = 1 + {popt2[0]:.3f}*(1 - exp(-R/{popt2[1]:.3f})) * "
                  f"exp(-ρ/10^{popt2[2]:.3f} / 10^{popt2[3]:.3f})")
        except:
            print("Double exponential fit failed")
        
        # Get NN parameters
        print(f"\nNeural network base parameters:")
        print(f"  ρ_c = 10^{self.model.rho_c.item():.3f} M☉/kpc³")
        print(f"  n = {self.model.n_exp.item():.3f}")
        print(f"  A = {self.model.A_boost.item():.3f}")
        
        return xi_mesh, R_mesh, rho_mesh

def visualize_results(engineer, model, trainer):
    """Create comprehensive visualizations."""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original rotation curve with fit
    ax = axes[0, 0]
    ax.scatter(engineer.R_data[::100], engineer.v_data[::100], alpha=0.3, s=1, label='Gaia data')
    
    R_plot = np.linspace(6, 18, 100)
    v_newton = engineer.calculate_newtonian_velocity(R_plot)
    ax.plot(R_plot, v_newton, 'b--', label='Newtonian')
    
    # Model prediction
    with torch.no_grad():
        R_tensor = torch.tensor(R_plot, dtype=torch.float32).to(device)
        rho_tensor = torch.tensor(engineer.calculate_density(R_plot), dtype=torch.float32).to(device)
        xi_pred = model(rho_tensor, R_tensor).cpu().numpy()
    
    v_model = v_newton * np.sqrt(xi_pred)
    ax.plot(R_plot, v_model, 'r-', linewidth=2, label='Reverse engineered')
    
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('V (km/s)')
    ax.set_title('Rotation Curve Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Xi enhancement vs R
    ax = axes[0, 1]
    ax.scatter(engineer.R_binned, engineer.xi_binned, s=50, label='Empirical ξ')
    ax.plot(R_plot, xi_pred, 'r-', linewidth=2, label='Model ξ')
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('ξ enhancement')
    ax.set_title('Gravity Enhancement Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Xi as function of density
    ax = axes[0, 2]
    rho_range = np.logspace(3, 21, 200)
    R_fixed = 8.0  # Solar radius
    
    with torch.no_grad():
        rho_tensor = torch.tensor(rho_range, dtype=torch.float32).to(device)
        R_tensor = torch.full_like(rho_tensor, R_fixed)
        xi_vs_rho = model(rho_tensor, R_tensor).cpu().numpy()
    
    ax.loglog(rho_range, xi_vs_rho - 1, 'r-', linewidth=2)
    ax.axvline(x=2.3e21, color='b', linestyle='--', label='Saturn orbit')
    ax.set_xlabel('ρ (M☉/kpc³)')
    ax.set_ylabel('ξ - 1')
    ax.set_title('Enhancement vs Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 2D heatmap of xi(R, ρ)
    ax = axes[1, 0]
    R_grid = np.linspace(1, 20, 50)
    rho_grid = np.logspace(3, 21, 50)
    R_mesh, rho_mesh = np.meshgrid(R_grid, rho_grid)
    
    with torch.no_grad():
        R_flat = torch.tensor(R_mesh.flatten(), dtype=torch.float32).to(device)
        rho_flat = torch.tensor(rho_mesh.flatten(), dtype=torch.float32).to(device)
        xi_flat = model(rho_flat, R_flat).cpu().numpy()
    
    xi_mesh = xi_flat.reshape(R_mesh.shape)
    
    im = ax.pcolormesh(R_mesh, rho_mesh, xi_mesh, shading='auto', cmap='viridis')
    ax.set_yscale('log')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('ρ (M☉/kpc³)')
    ax.set_title('ξ(R, ρ) Map')
    plt.colorbar(im, ax=ax, label='ξ')
    
    # 5. Training history
    ax = axes[1, 1]
    train_losses, val_losses = trainer.train_losses, trainer.val_losses
    ax.semilogy(train_losses, label='Train')
    ax.semilogy(val_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Residuals
    ax = axes[1, 2]
    v_residuals = engineer.v_data - np.interp(engineer.R_data, R_plot, v_model)
    ax.hexbin(engineer.R_data, v_residuals, gridsize=30, cmap='Blues')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('v_obs - v_model (km/s)')
    ax.set_title('Velocity Residuals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reverse_engineered_gravity.png', dpi=150)
    print("Saved visualizations to reverse_engineered_gravity.png")
    
    return fig

def main():
    """Main execution function."""
    print("="*60)
    print("REVERSE ENGINEERING GRAVITY FROM GAIA DATA")
    print("="*60)
    
    # Initialize
    engineer = GravityReverseEngineer()
    
    # Load data
    gaia_df = engineer.load_gaia_data()
    
    # Create model
    model = PhysicsInformedNN(hidden_layers=[128, 64, 32])
    
    # Initialize trainer
    trainer = GravityTrainer(engineer, model)
    trainer.prepare_data()
    
    # Train
    start_time = time.time()
    trainer.train_losses, trainer.val_losses = trainer.train(epochs=500, cassini_weight=1000.0)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")
    
    # Extract formula
    xi_mesh, R_mesh, rho_mesh = trainer.extract_formula()
    
    # Visualize
    fig = visualize_results(engineer, model, trainer)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'xi_mesh': xi_mesh,
        'R_mesh': R_mesh,
        'rho_mesh': rho_mesh,
        'baryon_params': engineer.baryon_params
    }, 'reverse_engineered_gravity_model.pt')
    
    print("\n" + "="*60)
    print("REVERSE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\nFinal parameters:")
    print(f"  ρ_c = 10^{model.rho_c.item():.3f} M☉/kpc³")
    print(f"  n = {model.n_exp.item():.3f}")
    print(f"  A = {model.A_boost.item():.3f}")
    
    # Check Cassini
    with torch.no_grad():
        rho_saturn = torch.tensor([2.3e21], device=device)
        R_saturn = torch.tensor([9.5e-3], device=device)  # AU to kpc
        xi_saturn = model(rho_saturn, R_saturn).item()
        print(f"\nCassini check: ξ(Saturn) = {xi_saturn:.8f} (deviation: {abs(xi_saturn-1):.2e})")
    
    # Generate formula code
    print("\nPython implementation of reverse-engineered gravity:")
    print("-"*60)
    print(f"""
def xi_reverse_engineered(rho, R, z=0):
    '''
    Reverse-engineered gravity enhancement factor.
    
    Parameters:
    - rho: density in M_sun/kpc³
    - R: galactocentric radius in kpc
    - z: height above disk in kpc
    '''
    # Neural network approximation
    log_rho = np.log10(rho + 1e-10)
    R_norm = R / 8.0
    z_norm = z / 0.5
    
    # Simplified analytical fit from NN
    rho_c = 10**{model.rho_c.item():.3f}
    n = {model.n_exp.item():.3f}
    A = {model.A_boost.item():.3f}
    
    # Core formula
    density_factor = 1 / (1 + (rho/rho_c)**n)
    radial_factor = 1 + 0.2 * (R/8.0 - 1)  # Empirical radial dependence
    vertical_factor = np.exp(-abs(z)/0.5)  # Disk confinement
    
    xi = 1 + A * density_factor * radial_factor * vertical_factor
    
    return xi
""")
    print("-"*60)

if __name__ == '__main__':
    main()
