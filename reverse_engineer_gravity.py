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
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: {device}")
    print("GPU: Apple M1 (MPS)")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print("GPU: CPU only")

class GravityReverseEngineer:
    """Main class for reverse engineering gravity from rotation curves."""
    
    def __init__(self, gaia_data_path='data/gaia_processed/gaia_processed_data.csv'):
        # Handle relative paths when running from different directories
        import os
        if not os.path.exists(gaia_data_path):
            # Try parent directory
            gaia_data_path = os.path.join('..', gaia_data_path)
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
        self.v_data = df_filtered['v_circ'].values  # Use circular velocity instead of v_obs
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

    def validate_model_physics(self):
            """Validate that the model captures key physics before formula extraction."""
            print("\n" + "="*60)
            print("VALIDATING MODEL PHYSICS")
            print("="*60)
            
            validation_passed = True
            metrics = {}
            
            # Test 1: Cassini constraint
            with torch.no_grad():
                rho_saturn = torch.tensor([2.3e21], device=device)
                R_saturn = torch.tensor([9.5e-6], device=device)  # AU in kpc
                xi_saturn = self.model(rho_saturn, R_saturn).item()
                cassini_deviation = abs(xi_saturn - 1.0)
                
            metrics['cassini_deviation'] = cassini_deviation
            print(f"\n1. Cassini constraint:")
            print(f"   ξ(Saturn) = {xi_saturn:.8f}")
            print(f"   Deviation = {cassini_deviation:.2e}")
            print(f"   Status: {'✓ PASS' if cassini_deviation < 1e-5 else '✗ FAIL'}")
            if cassini_deviation > 1e-5:
                validation_passed = False
                
            # Test 2: Galaxy edge enhancement
            with torch.no_grad():
                # Inner galaxy
                rho_inner = torch.tensor([self.engineer.calculate_density(6.0)], device=device)
                R_inner = torch.tensor([6.0], device=device)
                xi_inner = self.model(rho_inner, R_inner).item()
                
                # Solar neighborhood
                rho_solar = torch.tensor([self.engineer.calculate_density(8.0)], device=device)
                R_solar = torch.tensor([8.0], device=device)
                xi_solar = self.model(rho_solar, R_solar).item()
                
                # Galaxy edge
                rho_edge = torch.tensor([self.engineer.calculate_density(15.0)], device=device)
                R_edge = torch.tensor([15.0], device=device)
                xi_edge = self.model(rho_edge, R_edge).item()
                
            metrics['xi_inner'] = xi_inner
            metrics['xi_solar'] = xi_solar
            metrics['xi_edge'] = xi_edge
            
            print(f"\n2. Galactic gradient:")
            print(f"   ξ(6 kpc) = {xi_inner:.3f}")
            print(f"   ξ(8 kpc) = {xi_solar:.3f}")
            print(f"   ξ(15 kpc) = {xi_edge:.3f}")
            
            # Check for proper enhancement gradient
            enhancement_gradient = xi_edge > xi_solar > xi_inner > 1.0
            min_edge_enhancement = (xi_edge - 1.0) > 0.5  # At least 50% enhancement at edge
            
            print(f"   Gradient check: {'✓ PASS' if enhancement_gradient else '✗ FAIL'}")
            print(f"   Edge enhancement: {(xi_edge-1)*100:.1f}% {'✓ PASS' if min_edge_enhancement else '✗ FAIL (need >50%)'}")
            
            if not enhancement_gradient or not min_edge_enhancement:
                validation_passed = False
                
            # Test 3: Rotation curve fit quality
            print(f"\n3. Rotation curve fit:")
            
            # Calculate predicted vs observed velocities
            R_test = np.linspace(6, 18, 50)
            v_newton = self.engineer.calculate_newtonian_velocity(R_test)
            
            with torch.no_grad():
                R_tensor = torch.tensor(R_test, dtype=torch.float32).to(device)
                rho_tensor = torch.tensor(self.engineer.calculate_density(R_test), dtype=torch.float32).to(device)
                xi_pred = self.model(rho_tensor, R_tensor).cpu().numpy()
            
            v_model = v_newton * np.sqrt(xi_pred)
            
            # Compare to empirical data in same range
            mask = (self.engineer.R_data >= 6) & (self.engineer.R_data <= 18)
            R_emp = self.engineer.R_data[mask]
            v_emp = self.engineer.v_data[mask]
            
            # Interpolate model to empirical points
            v_model_at_data = np.interp(R_emp, R_test, v_model)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((v_model_at_data - v_emp)**2))
            relative_error = np.mean(np.abs(v_model_at_data - v_emp) / v_emp) * 100
            
            metrics['velocity_rmse'] = rmse
            metrics['velocity_relative_error'] = relative_error
            
            print(f"   RMSE = {rmse:.1f} km/s")
            print(f"   Mean relative error = {relative_error:.1f}%")
            print(f"   Status: {'✓ PASS' if relative_error < 10 else '✗ FAIL (need <10%)'}")
            
            if relative_error > 10:
                validation_passed = False
                
            # Test 4: Check if model learned non-trivial solution
            print(f"\n4. Non-trivial solution check:")
            
            # Check variance in xi across parameter space
            with torch.no_grad():
                R_sample = torch.tensor(np.random.uniform(5, 20, 100), dtype=torch.float32).to(device)
                rho_sample = torch.tensor(10**np.random.uniform(9, 13, 100), dtype=torch.float32).to(device)
                xi_sample = self.model(rho_sample, R_sample).cpu().numpy()
            
            xi_variance = np.var(xi_sample)
            xi_range = np.max(xi_sample) - np.min(xi_sample)
            
            metrics['xi_variance'] = xi_variance
            metrics['xi_range'] = xi_range
            
            print(f"   ξ variance = {xi_variance:.4f}")
            print(f"   ξ range = {xi_range:.3f}")
            print(f"   Status: {'✓ PASS' if xi_variance > 0.01 else '✗ FAIL (model is too flat!)'}")
            
            if xi_variance < 0.01:
                validation_passed = False
                print("\n   ⚠️  Model has converged to trivial solution (ξ ≈ constant)")
                
            # Overall assessment
            print("\n" + "="*60)
            print(f"OVERALL VALIDATION: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
            print("="*60)
            
            if not validation_passed:
                print("\n⚠️  Model needs more training or parameter tuning!")
                print("Recommendations:")
                if cassini_deviation > 1e-5:
                    print("  - Increase Cassini constraint weight")
                if not enhancement_gradient:
                    print("  - Check learning rate and network architecture")
                if xi_variance < 0.01:
                    print("  - Model may be stuck in local minimum")
                    print("  - Try different initialization or learning rate schedule")
                if relative_error > 10:
                    print("  - Train for more epochs")
                    print("  - Adjust loss function weights")
                    
            return validation_passed, metrics
    
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

        
    def extract_physics_formulas(self):
        """Extract multiple candidate gravity formulas from the trained model."""
        print("\n" + "="*60)
        print("EXTRACTING CANDIDATE GRAVITY FORMULAS")
        print("="*60)
        
        formulas = []
        
        # Get neural network parameters
        with torch.no_grad():
            rho_c = 10**self.model.rho_c.item()
            n = self.model.n_exp.item()
            A = self.model.A_boost.item()
        
        print(f"\nBase NN parameters: ρ_c = {rho_c:.2e} M☉/kpc³, n = {n:.3f}, A = {A:.3f}")
        
        # Generate test data covering wide range
        R_test = np.logspace(-3, 2, 200)  # 0.001 to 100 kpc (includes solar system)
        rho_test = np.logspace(3, 23, 200)  # Wide density range
        
        # Test specific regions
        regions = {
            'Solar System': {'R': np.array([1e-5, 1e-4, 1e-3]), 'rho': np.array([1e21, 2.3e21, 1e22])},
            'Galactic': {'R': np.linspace(5, 20, 50), 'rho': np.logspace(9, 13, 50)},
            'Extreme': {'R': np.logspace(-6, 3, 100), 'rho': np.logspace(3, 25, 100)}
        }
        
        # Evaluate model in each region
        xi_data = {}
        for region_name, region_data in regions.items():
            R_mesh, rho_mesh = np.meshgrid(region_data['R'], region_data['rho'])
            with torch.no_grad():
                R_flat = torch.tensor(R_mesh.flatten(), dtype=torch.float32).to(device)
                rho_flat = torch.tensor(rho_mesh.flatten(), dtype=torch.float32).to(device)
                xi_flat = self.model(rho_flat, R_flat).cpu().numpy()
            xi_data[region_name] = (R_mesh, rho_mesh, xi_flat.reshape(R_mesh.shape))
        
        # Formula 1: Modified MOND-like
        print("\n1. Modified MOND-like formula:")
        a0_eff = 1.2e-10  # m/s²
        formula1 = {
            'name': 'Modified MOND',
            'formula': lambda rho, R: 1 + A * (1 - 1/(1 + (R * a0_eff * 3.086e16 / (4.302e-6 * rho * R**2))**0.5)),
            'params': {'a0_eff': a0_eff, 'A': A},
            'description': 'ξ = 1 + A * (1 - 1/(1 + (a/a0)^0.5))'
        }
        formulas.append(formula1)
        
        # Formula 2: Yukawa-like screening
        print("\n2. Yukawa-like screening formula:")
        lambda_screen = 10**((np.log10(rho_c) - 21) / 2)  # Screening length in kpc
        formula2 = {
            'name': 'Yukawa Screening',
            'formula': lambda rho, R: 1 + A * np.exp(-R/lambda_screen) / (1 + (rho/rho_c)**n),
            'params': {'lambda_screen': lambda_screen, 'rho_c': rho_c, 'n': n, 'A': A},
            'description': f'ξ = 1 + {A:.2f} * exp(-R/{lambda_screen:.2e}) / (1 + (ρ/{rho_c:.2e})^{n:.2f})'
        }
        formulas.append(formula2)
        
        # Formula 3: Chameleon-like
        print("\n3. Chameleon-like formula:")
        M_pl = 2.4e18  # GeV
        formula3 = {
            'name': 'Chameleon',
            'formula': lambda rho, R: 1 + A * (1 - np.tanh((rho/rho_c)**0.5)),
            'params': {'rho_c': rho_c, 'A': A},
            'description': f'ξ = 1 + {A:.2f} * (1 - tanh((ρ/{rho_c:.2e})^0.5))'
        }
        formulas.append(formula3)
        
        # Formula 4: f(R) gravity inspired
        print("\n4. f(R) gravity inspired formula:")
        R_curv_0 = 1e-30  # Curvature scale
        formula4 = {
            'name': 'f(R) Inspired',
            'formula': lambda rho, R: 1 + A * R**2 / (R**2 + (rho/rho_c)**(-1/3)),
            'params': {'rho_c': rho_c, 'A': A},
            'description': f'ξ = 1 + {A:.2f} * R² / (R² + (ρ/{rho_c:.2e})^(-1/3))'
        }
        formulas.append(formula4)
        
        # Formula 5: Emergent gravity
        print("\n5. Emergent gravity formula:")
        a_D = 1e-10  # de Sitter acceleration
        formula5 = {
            'name': 'Emergent Gravity',
            'formula': lambda rho, R: 1 + A * np.sqrt(1 + 4/(1 + (rho/rho_c)**2)) - 1,
            'params': {'rho_c': rho_c, 'A': A, 'a_D': a_D},
            'description': f'ξ = 1 + {A:.2f} * (√(1 + 4/(1 + (ρ/{rho_c:.2e})²)) - 1)'
        }
        formulas.append(formula5)
        
        # Validate each formula
        print("\n" + "="*60)
        print("VALIDATING CANDIDATE FORMULAS")
        print("="*60)
        
        for formula in formulas:
            print(f"\n{formula['name']}:")
            print(f"  {formula['description']}")
            
            # Test Cassini constraint
            rho_saturn = 2.3e21  # M☉/kpc³
            R_saturn = 9.5e-6  # kpc (9.5 AU)
            xi_cassini = formula['formula'](rho_saturn, R_saturn)
            cassini_deviation = abs(xi_cassini - 1.0)
            
            # Test galactic edge
            rho_edge = 1e10  # Typical edge density
            R_edge = 15.0  # kpc
            xi_edge = formula['formula'](rho_edge, R_edge)
            
            print(f"  Cassini: ξ = {xi_cassini:.8f} (deviation: {cassini_deviation:.2e})")
            print(f"  Galactic edge: ξ = {xi_edge:.3f} (enhancement: {(xi_edge-1)*100:.1f}%)")
            
            # Score based on matching NN predictions
            score = 0
            with torch.no_grad():
                test_points = 100
                R_sample = np.random.uniform(5, 20, test_points)
                rho_sample = 10**np.random.uniform(9, 13, test_points)
                
                R_tensor = torch.tensor(R_sample, dtype=torch.float32).to(device)
                rho_tensor = torch.tensor(rho_sample, dtype=torch.float32).to(device)
                xi_nn = self.model(rho_tensor, R_tensor).cpu().numpy()
                xi_formula = formula['formula'](rho_sample, R_sample)
                
                mse = np.mean((xi_nn - xi_formula)**2)
                score = 1 / (1 + mse)
            
            print(f"  NN match score: {score:.3f}")
            formula['score'] = score
            formula['cassini_ok'] = cassini_deviation < 1e-5
        
        # Sort by score
        formulas.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate LaTeX
        print("\n" + "="*60)
        print("LATEX FORMULATIONS FOR PAPER")
        print("="*60)
        
        for i, formula in enumerate(formulas[:3]):  # Top 3
            if formula['cassini_ok']:
                print(f"\n{i+1}. {formula['name']} (Score: {formula['score']:.3f}) ✓")
                print("LaTeX:")
                if formula['name'] == 'Modified MOND':
                    print(r"  \xi = 1 + A \left(1 - \frac{1}{1 + \sqrt{a/a_0}}\right)")
                elif formula['name'] == 'Yukawa Screening':
                    print(r"  \xi = 1 + A \frac{e^{-R/\lambda}}{1 + (\rho/\rho_c)^n}")
                elif formula['name'] == 'Chameleon':
                    print(r"  \xi = 1 + A \left(1 - \tanh\sqrt{\rho/\rho_c}\right)")
                elif formula['name'] == 'f(R) Inspired':
                    print(r"  \xi = 1 + A \frac{R^2}{R^2 + (\rho/\rho_c)^{-1/3}}")
                elif formula['name'] == 'Emergent Gravity':
                    print(r"  \xi = 1 + A \left(\sqrt{1 + \frac{4}{1 + (\rho/\rho_c)^2}} - 1\right)")
        
        return formulas


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
    ax.set_ylabel('v_circ - v_model (km/s)')
    ax.set_title('Velocity Residuals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/reverse_engineered_gravity.png', dpi=150)
    print("Saved visualizations to plots/reverse_engineered_gravity.png")
    
    return fig

def visualize_formulas(formulas, engineer, model):
    """Visualize candidate gravity formulas."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Test ranges
    R_galactic = np.linspace(5, 20, 100)
    R_solar = np.logspace(-6, -3, 100)  # AU scale in kpc
    rho_galactic = engineer.calculate_density(R_galactic)
    rho_solar = np.full_like(R_solar, 2.3e21)  # Solar system density
    
    # 1. Galactic scale comparison
    ax = axes[0, 0]
    for formula in formulas[:3]:
        if formula['cassini_ok']:
            xi_gal = formula['formula'](rho_galactic, R_galactic)
            ax.plot(R_galactic, xi_gal, label=formula['name'], linewidth=2)
    
    # Add NN prediction
    with torch.no_grad():
        R_tensor = torch.tensor(R_galactic, dtype=torch.float32).to(device)
        rho_tensor = torch.tensor(rho_galactic, dtype=torch.float32).to(device)
        xi_nn = model(rho_tensor, R_tensor).cpu().numpy()
    ax.plot(R_galactic, xi_nn, 'k--', label='Neural Network', linewidth=2)
    
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('ξ')
    ax.set_title('Galactic Scale Enhancement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Solar system scale
    ax = axes[0, 1]
    for formula in formulas[:3]:
        if formula['cassini_ok']:
            xi_solar = formula['formula'](rho_solar, R_solar)
            deviation = (xi_solar - 1) * 1e6  # Parts per million
            ax.semilogx(R_solar * 206265, deviation, label=formula['name'], linewidth=2)
    
    ax.axhline(y=23, color='r', linestyle='--', label='Cassini limit')
    ax.axhline(y=-23, color='r', linestyle='--')
    ax.set_xlabel('R (AU)')
    ax.set_ylabel('(ξ - 1) × 10⁶')
    ax.set_title('Solar System Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Density dependence
    ax = axes[0, 2]
    rho_range = np.logspace(5, 22, 200)
    R_fixed = 8.0  # Solar radius
    
    for formula in formulas[:3]:
        if formula['cassini_ok']:
            xi_rho = formula['formula'](rho_range, R_fixed)
            ax.loglog(rho_range, xi_rho - 1, label=formula['name'], linewidth=2)
    
    ax.axvline(x=2.3e21, color='b', linestyle=':', label='Saturn')
    ax.set_xlabel('ρ (M☉/kpc³)')
    ax.set_ylabel('ξ - 1')
    ax.set_title('Density Screening')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Acceleration scale
    ax = axes[1, 0]
    for formula in formulas[:3]:
        if formula['cassini_ok']:
            # Calculate effective acceleration
            a_newton = 4.302e-6 * rho_galactic * R_galactic  # GM/R²
            xi_gal = formula['formula'](rho_galactic, R_galactic)
            a_eff = a_newton * xi_gal
            ax.loglog(a_newton * 3.086e13 / 3.154e7, xi_gal, label=formula['name'], linewidth=2)
    
    ax.axvline(x=1.2e-10, color='g', linestyle='--', label='a₀ (MOND)')
    ax.set_xlabel('a_Newton (m/s²)')
    ax.set_ylabel('ξ')
    ax.set_title('Enhancement vs Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Parameter space
    ax = axes[1, 1]
    R_mesh, rho_mesh = np.meshgrid(np.linspace(5, 20, 50), np.logspace(9, 13, 50))
    
    best_formula = formulas[0]
    xi_mesh = best_formula['formula'](rho_mesh, R_mesh)
    
    im = ax.pcolormesh(R_mesh, rho_mesh, xi_mesh, shading='auto', cmap='viridis')
    ax.set_yscale('log')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('ρ (M☉/kpc³)')
    ax.set_title(f"Best Formula: {best_formula['name']}")
    plt.colorbar(im, ax=ax, label='ξ')
    
    # 6. Residuals
    ax = axes[1, 2]
    with torch.no_grad():
        test_R = np.random.uniform(6, 18, 1000)
        test_rho = engineer.calculate_density(test_R)
        
        R_tensor = torch.tensor(test_R, dtype=torch.float32).to(device)
        rho_tensor = torch.tensor(test_rho, dtype=torch.float32).to(device)
        xi_nn = model(rho_tensor, R_tensor).cpu().numpy()
        
        xi_formula = best_formula['formula'](test_rho, test_R)
        residuals = (xi_formula - xi_nn) / xi_nn * 100
        
    ax.hexbin(test_R, residuals, gridsize=30, cmap='RdBu', vmin=-10, vmax=10)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Formula vs NN (%)')
    ax.set_title(f"{best_formula['name']} Residuals")
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/gravity_formulas_comparison.png', dpi=150)
    print("\nSaved formula comparisons to plots/gravity_formulas_comparison.png")
    
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
    trainer.train_losses, trainer.val_losses = trainer.train(epochs=50, cassini_weight=1000.0)
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
    }, 'data/reverse_engineered_gravity_model.pt')
    
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
