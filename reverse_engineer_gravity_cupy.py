#!/usr/bin/env python3
"""
reverse_engineer_gravity_cupy.py

Reverse engineer the formula for gravity from Gaia rotation curve data
using CuPy GPU-accelerated machine learning.

Requirements:
- CuPy with CUDA support
- numpy, pandas, matplotlib
- scikit-learn
"""

import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import time
from pathlib import Path
import os

# Configure CuPy for GPU
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")
print(f"Using device: {cp.cuda.Device(0)}")

# Set random seed for reproducibility
cp.random.seed(42)

class GravityReverseEngineer:
    """Main class for reverse engineering gravity from rotation curves."""
    
    def __init__(self, gaia_data_path='data/gaia_processed/gaia_processed_data.csv'):
        # Handle relative paths when running from different directories
        if not os.path.exists(gaia_data_path):
            # Try parent directory
            gaia_data_path = os.path.join('..', gaia_data_path)
        self.gaia_data_path = gaia_data_path
        
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

class PhysicsInformedNN:
    """Neural network that learns xi(ρ, R) with physical constraints."""
    
    def __init__(self, hidden_layers=[64, 64, 32]):
        self.hidden_layers = hidden_layers
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        input_size = 3
        for hidden_size in hidden_layers:
            # Xavier initialization
            w = cp.random.normal(0, cp.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(cp.float32)
            b = cp.zeros(hidden_size, dtype=cp.float32)
            self.weights.append(w)
            self.biases.append(b)
            input_size = hidden_size
        
        # Output layer
        w_out = cp.random.normal(0, cp.sqrt(2.0 / input_size), (input_size, 1)).astype(cp.float32)
        b_out = cp.zeros(1, dtype=cp.float32)
        self.weights.append(w_out)
        self.biases.append(b_out)
        
        # Learnable parameters for analytical constraints
        self.rho_c = cp.array([12.0], dtype=cp.float32)  # log10(rho_c)
        self.n_exp = cp.array([1.5], dtype=cp.float32)
        self.A_boost = cp.array([2.0], dtype=cp.float32)
        
    def forward(self, rho, R, z=None):
        """
        Forward pass computing xi.
        
        Parameters:
        - rho: density in M_sun/kpc³
        - R: galactocentric radius in kpc
        - z: height above plane in kpc
        """
        if z is None:
            z = cp.zeros_like(R)
        
        # Prepare inputs (normalize and log-transform)
        log_rho = cp.log10(rho + 1e-10)
        R_norm = R / 8.0  # Normalize by R_sun
        z_norm = z / 0.5  # Normalize by scale height
        
        x = cp.stack([log_rho, R_norm, z_norm], axis=-1)
        
        # Forward pass through hidden layers
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            x = cp.dot(x, w) + b
            x = cp.maximum(0, x)  # ReLU activation
            if i < len(self.weights) - 2:  # Dropout on hidden layers
                x = cp.where(cp.random.random(x.shape) > 0.1, x, 0)
        
        # Output layer
        nn_output = cp.dot(x, self.weights[-1]) + self.biases[-1]
        nn_output = nn_output.squeeze()
        
        # Physics-based modulation
        rho_ratio = rho / (10**self.rho_c[0])
        density_factor = 1 / (1 + rho_ratio**self.n_exp[0])
        
        # Combine NN with physics
        xi = 1 + self.A_boost[0] * cp.sigmoid(nn_output) * density_factor
        
        return xi
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        params.extend([self.rho_c, self.n_exp, self.A_boost])
        return params
    
    def set_parameters(self, params):
        """Set all trainable parameters."""
        idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.weights[i] = params[idx]
            self.biases[i] = params[idx + 1]
            idx += 2
        self.rho_c = params[idx]
        self.n_exp = params[idx + 1]
        self.A_boost = params[idx + 2]

class GravityTrainer:
    """Train the physics-informed neural network."""
    
    def __init__(self, engineer, model):
        self.engineer = engineer
        self.model = model
        
        # Optimizer parameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        
    def prepare_data(self):
        """Prepare training data."""
        # Get empirical xi values
        R_binned, xi_binned = self.engineer.derive_empirical_xi()
        
        # Prepare full dataset
        self.R_data = cp.array(self.engineer.R_clean, dtype=cp.float32)
        self.rho_data = cp.array(self.engineer.rho_clean, dtype=cp.float32)
        self.xi_data = cp.array(self.engineer.xi_clean, dtype=cp.float32)
        
        # Split train/val
        n_train = int(0.8 * len(self.R_data))
        indices = cp.arange(len(self.R_data))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        self.train_data = {
            'R': self.R_data[train_indices],
            'rho': self.rho_data[train_indices],
            'xi': self.xi_data[train_indices]
        }
        
        self.val_data = {
            'R': self.R_data[val_indices],
            'rho': self.rho_data[val_indices],
            'xi': self.xi_data[val_indices]
        }
        
        return len(train_indices), len(val_indices)

    def compute_loss(self, rho, R, xi_target):
        """Compute loss function."""
        # Forward pass
        xi_pred = self.model.forward(rho, R)
        
        # MSE loss
        mse_loss = cp.mean((xi_pred - xi_target)**2)
        
        # Cassini constraint
        rho_saturn = cp.array([2.3e21], dtype=cp.float32)
        R_saturn = cp.array([9.5e-6], dtype=cp.float32)
        xi_saturn = self.model.forward(rho_saturn, R_saturn)
        cassini_loss = (xi_saturn - 1.0)**2 / (2.3e-5)**2
        
        # Physical regularization
        reg_loss = 0.01 * (cp.abs(self.model.n_exp - 1.5) + 
                           cp.abs(self.model.A_boost - 2.0))
        
        # Total loss
        total_loss = mse_loss + 100.0 * cassini_loss + reg_loss
        
        return total_loss, mse_loss, cassini_loss, reg_loss

    def compute_gradients(self, rho, R, xi_target):
        """Compute gradients using CuPy's automatic differentiation."""
        # CuPy doesn't have automatic differentiation like JAX
        # We'll use finite differences for gradients
        params = self.model.get_parameters()
        gradients = []
        
        epsilon = 1e-6
        for param in params:
            grad = cp.zeros_like(param)
            
            # Compute gradients using finite differences
            for i in range(param.size):
                # Save original value
                original_val = param.flat[i].copy()
                
                # Forward pass with param + epsilon
                param.flat[i] = original_val + epsilon
                self.model.set_parameters(params)
                loss_plus, _, _, _ = self.compute_loss(rho, R, xi_target)
                
                # Forward pass with param - epsilon
                param.flat[i] = original_val - epsilon
                self.model.set_parameters(params)
                loss_minus, _, _, _ = self.compute_loss(rho, R, xi_target)
                
                # Restore original value
                param.flat[i] = original_val
                
                # Compute gradient
                grad.flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            gradients.append(grad)
        
        return gradients

    def update_parameters(self, gradients):
        """Update parameters using gradients."""
        params = self.model.get_parameters()
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Adam-like update
            param -= self.learning_rate * grad

    def validate_model_physics(self):
        """Validate that the model captures key physics before formula extraction."""
        print("\n" + "="*60)
        print("VALIDATING MODEL PHYSICS")
        print("="*60)
        
        validation_passed = True
        metrics = {}
        
        # Test 1: Cassini constraint
        rho_saturn = cp.array([2.3e21], dtype=cp.float32)
        R_saturn = cp.array([9.5e-6], dtype=cp.float32)  # AU in kpc
        xi_saturn = self.model.forward(rho_saturn, R_saturn).get()
        cassini_deviation = abs(xi_saturn - 1.0)
        
        metrics['cassini_deviation'] = cassini_deviation
        print(f"\n1. Cassini constraint:")
        print(f"   ξ(Saturn) = {xi_saturn:.8f}")
        print(f"   Deviation = {cassini_deviation:.2e}")
        print(f"   Status: {'✓ PASS' if cassini_deviation < 1e-5 else '✗ FAIL'}")
        if cassini_deviation > 1e-5:
            validation_passed = False
            
        # Test 2: Galaxy edge enhancement
        # Inner galaxy
        rho_inner = cp.array([self.engineer.calculate_density(6.0)], dtype=cp.float32)
        R_inner = cp.array([6.0], dtype=cp.float32)
        xi_inner = self.model.forward(rho_inner, R_inner).get()
        
        # Solar neighborhood
        rho_solar = cp.array([self.engineer.calculate_density(8.0)], dtype=cp.float32)
        R_solar = cp.array([8.0], dtype=cp.float32)
        xi_solar = self.model.forward(rho_solar, R_solar).get()
        
        # Galaxy edge
        rho_edge = cp.array([self.engineer.calculate_density(15.0)], dtype=cp.float32)
        R_edge = cp.array([15.0], dtype=cp.float32)
        xi_edge = self.model.forward(rho_edge, R_edge).get()
        
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
        
        R_tensor = cp.array(R_test, dtype=cp.float32)
        rho_tensor = cp.array(self.engineer.calculate_density(R_test), dtype=cp.float32)
        xi_pred = self.model.forward(rho_tensor, R_tensor).get()
        
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
        R_sample = cp.array(np.random.uniform(5, 20, 100), dtype=cp.float32)
        rho_sample = cp.array(10**np.random.uniform(9, 13, 100), dtype=cp.float32)
        xi_sample = self.model.forward(rho_sample, R_sample).get()
        
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
    
    def train(self, epochs=5000):
        """Train the model."""
        print(f"\nTraining on CuPy GPU for {epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            batch_size = min(1024, len(self.train_data['R']))
            indices = cp.random.permutation(len(self.train_data['R']))[:batch_size]
            
            rho_batch = self.train_data['rho'][indices]
            R_batch = self.train_data['R'][indices]
            xi_batch = self.train_data['xi'][indices]
            
            # Compute gradients
            gradients = self.compute_gradients(rho_batch, R_batch, xi_batch)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Compute losses
            train_loss, _, _, _ = self.compute_loss(rho_batch, R_batch, xi_batch)
            val_loss, _, _, _ = self.compute_loss(
                self.val_data['rho'], self.val_data['R'], self.val_data['xi']
            )
            
            train_losses.append(float(train_loss.get()))
            val_losses.append(float(val_loss.get()))
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.get():.4f}, Val Loss = {val_loss.get():.4f}")
                
                # Get current parameters
                rho_c = self.model.rho_c[0].get()
                n_exp = self.model.n_exp[0].get()
                A_boost = self.model.A_boost[0].get()
                
                print(f"  Parameters: rho_c = 10^{rho_c:.2f}, "
                      f"n = {n_exp:.2f}, A = {A_boost:.2f}")
                
                # Check Cassini
                rho_saturn = cp.array([2.3e21], dtype=cp.float32)
                R_saturn = cp.array([9.5e-6], dtype=cp.float32)
                xi_saturn = self.model.forward(rho_saturn, R_saturn)
                cassini_loss = (xi_saturn - 1.0)**2 / (2.3e-5)**2
                print(f"  Cassini violation: {cassini_loss.get():.2e}")
        
        # Store for later access
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        return train_losses, val_losses

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
    R_tensor = cp.array(R_plot, dtype=cp.float32)
    rho_tensor = cp.array(engineer.calculate_density(R_plot), dtype=cp.float32)
    xi_pred = model.forward(rho_tensor, R_tensor).get()
    
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
    
    rho_tensor = cp.array(rho_range, dtype=cp.float32)
    R_tensor = cp.full_like(rho_tensor, R_fixed)
    xi_vs_rho = model.forward(rho_tensor, R_tensor).get()
    
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
    
    R_flat = cp.array(R_mesh.flatten(), dtype=cp.float32)
    rho_flat = cp.array(rho_mesh.flatten(), dtype=cp.float32)
    xi_flat = model.forward(rho_flat, R_flat).get()
    
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
    plt.savefig('plots/reverse_engineered_gravity_cupy.png', dpi=150)
    print("Saved visualizations to plots/reverse_engineered_gravity_cupy.png")
    
    return fig

def main():
    """Main execution function."""
    print("="*60)
    print("REVERSE ENGINEERING GRAVITY FROM GAIA DATA (CuPy GPU)")
    print("="*60)
    
    # Initialize
    engineer = GravityReverseEngineer()
    
    # Load data
    gaia_df = engineer.load_gaia_data()
    
    # Create model
    model = PhysicsInformedNN(hidden_layers=[128, 64, 32])
    
    # Initialize trainer
    trainer = GravityTrainer(engineer, model)
    n_train, n_val = trainer.prepare_data()
    print(f"Training on {n_train} samples, validating on {n_val} samples")
    
    # Train
    start_time = time.time()
    trainer.train_losses, trainer.val_losses = trainer.train(epochs=5000)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")
    
    # Validate model physics
    validation_passed, validation_metrics = trainer.validate_model_physics()
    
    if not validation_passed:
        print("\n" + "="*60)
        print("⚠️  MODEL VALIDATION FAILED!")
        print("="*60)
        print("\nThe model has not learned the correct physics.")
        print("Suggestions:")
        print("1. Train for more epochs (try 10000-20000)")
        print("2. Adjust learning rate (current: 1e-3)")
        print("3. Check data preprocessing")
        print("4. Modify network architecture")
        print("\nContinuing with visualization but skipping formula extraction...")
        
        # Still visualize to see what went wrong
        fig = visualize_results(engineer, model, trainer)
        
    else:
        print("\n✓ Model validation passed! Proceeding with visualization...")
        
        # Visualize
        fig = visualize_results(engineer, model, trainer)
        
        # Save complete model
        import pickle
        model_data = {
            'model': model,
            'baryon_params': engineer.baryon_params,
            'validation_metrics': validation_metrics,
            'epochs_trained': 5000,
            'status': 'validated'
        }
        
        with open('data/reverse_engineered_gravity_model_cupy.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    print("\n" + "="*60)
    print("REVERSE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\nFinal parameters:")
    rho_c = model.rho_c[0].get()
    n_exp = model.n_exp[0].get()
    A_boost = model.A_boost[0].get()
    print(f"  ρ_c = 10^{rho_c:.3f} M☉/kpc³")
    print(f"  n = {n_exp:.3f}")
    print(f"  A = {A_boost:.3f}")
    
    # Check Cassini
    rho_saturn = cp.array([2.3e21], dtype=cp.float32)
    R_saturn = cp.array([9.5e-3], dtype=cp.float32)  # AU to kpc
    xi_saturn = model.forward(rho_saturn, R_saturn).get()
    print(f"\nCassini check: ξ(Saturn) = {xi_saturn:.8f} (deviation: {abs(xi_saturn-1):.2e})")

if __name__ == '__main__':
    main() 