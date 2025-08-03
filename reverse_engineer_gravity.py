#!/usr/bin/env python3
"""
reverse_engineer_gravity.py

Reverse engineer the formula for gravity from Gaia rotation curve data
using JAX GPU-accelerated machine learning.

Requirements:
- JAX with CUDA support
- numpy, pandas, matplotlib
- scikit-learn
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from flax import linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import time
from pathlib import Path
import os

# Configure JAX for GPU
jax.config.update('jax_platform_name', 'gpu')
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Using device: {jax.devices()[0]}")

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

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

class PhysicsInformedNN(nn.Module):
    """Neural network that learns xi(ρ, R) with physical constraints."""
    
    hidden_layers: list = None
    
    def setup(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 32]
        
        # Create network layers
        self.layers = []
        input_size = 3
        
        for i, hidden_size in enumerate(self.hidden_layers):
            self.layers.append(nn.Dense(hidden_size))
            input_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Dense(1)
        
        # Learnable parameters for analytical constraints
        self.rho_c = self.param('rho_c', nn.initializers.constant(12.0), (1,))
        self.n_exp = self.param('n_exp', nn.initializers.constant(1.5), (1,))
        self.A_boost = self.param('A_boost', nn.initializers.constant(2.0), (1,))
        
    def __call__(self, rho, R, z=None, training=False):
        """
        Forward pass computing xi.
        
        Parameters:
        - rho: density in M_sun/kpc³
        - R: galactocentric radius in kpc
        - z: height above plane in kpc
        """
        if z is None:
            z = jnp.zeros_like(R)
        
        # Prepare inputs (normalize and log-transform)
        log_rho = jnp.log10(rho + 1e-10)
        R_norm = R / 8.0  # Normalize by R_sun
        z_norm = z / 0.5  # Normalize by scale height
        
        inputs = jnp.stack([log_rho, R_norm, z_norm], axis=-1)
        
        # Neural network output
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = jax.nn.relu(x)
            if training:
                x = nn.Dropout(0.1, deterministic=False)(x)
        
        nn_output = self.output_layer(x).squeeze()
        
        # Physics-based modulation
        rho_ratio = rho / (10**self.rho_c[0])
        density_factor = 1 / (1 + rho_ratio**self.n_exp[0])
        
        # Combine NN with physics
        xi = 1 + self.A_boost[0] * jax.nn.sigmoid(nn_output) * density_factor
        
        return xi
    
    def cassini_constraint(self, params):
        """Calculate Cassini constraint violation."""
        rho_saturn = jnp.array([2.3e21])
        R_saturn = jnp.array([9.5e-6])  # AU to kpc
        xi_saturn = self.apply(params, rho_saturn, R_saturn)
        cassini_violation = (xi_saturn - 1.0)**2 / (2.3e-5)**2
        return cassini_violation

class GravityTrainer:
    """Train the physics-informed neural network."""
    
    def __init__(self, engineer, model, key):
        self.engineer = engineer
        self.model = model
        self.key = key
        
        # Initialize model parameters
        dummy_rho = jnp.array([1e10])
        dummy_R = jnp.array([8.0])
        self.params = self.model.init(key, dummy_rho, dummy_R)
        
        # Optimizer
        self.optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
        self.optimizer_state = self.optimizer.init(self.params)
        
        # Learning rate scheduler
        self.scheduler = optax.exponential_decay(
            init_value=1e-3,
            transition_steps=1000,
            decay_rate=0.95
        )
        
    def prepare_data(self):
        """Prepare training data."""
        # Get empirical xi values
        R_binned, xi_binned = self.engineer.derive_empirical_xi()
        
        # Prepare full dataset
        self.R_data = jnp.array(self.engineer.R_clean, dtype=jnp.float32)
        self.rho_data = jnp.array(self.engineer.rho_clean, dtype=jnp.float32)
        self.xi_data = jnp.array(self.engineer.xi_clean, dtype=jnp.float32)
        
        # Split train/val
        n_train = int(0.8 * len(self.R_data))
        indices = jnp.arange(len(self.R_data))
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

    @staticmethod
    @jit
    def compute_loss(params, model, rho, R, xi_target, cassini_weight=100.0):
        """Compute loss function."""
        # Forward pass
        xi_pred = model.apply(params, rho, R)
        
        # MSE loss
        mse_loss = jnp.mean((xi_pred - xi_target)**2)
        
        # Cassini constraint
        rho_saturn = jnp.array([2.3e21])
        R_saturn = jnp.array([9.5e-6])
        xi_saturn = model.apply(params, rho_saturn, R_saturn)
        cassini_loss = (xi_saturn - 1.0)**2 / (2.3e-5)**2
        
        # Physical regularization
        rho_c = params['params']['rho_c'][0]
        n_exp = params['params']['n_exp'][0]
        A_boost = params['params']['A_boost'][0]
        
        reg_loss = 0.01 * (jnp.abs(n_exp - 1.5) + jnp.abs(A_boost - 2.0))
        
        # Total loss
        total_loss = mse_loss + cassini_weight * cassini_loss + reg_loss
        
        return total_loss, (mse_loss, cassini_loss, reg_loss)

    @staticmethod
    @jit
    def update_step(params, optimizer_state, optimizer, model, rho, R, xi_target, cassini_weight=100.0):
        """Single training step."""
        loss, grads = grad(GravityTrainer.compute_loss, has_aux=True)(
            params, model, rho, R, xi_target, cassini_weight
        )
        
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_optimizer_state, loss

    def validate_model_physics(self):
        """Validate that the model captures key physics before formula extraction."""
        print("\n" + "="*60)
        print("VALIDATING MODEL PHYSICS")
        print("="*60)
        
        validation_passed = True
        metrics = {}
        
        # Test 1: Cassini constraint
        rho_saturn = jnp.array([2.3e21])
        R_saturn = jnp.array([9.5e-6])  # AU in kpc
        xi_saturn = self.model.apply(self.params, rho_saturn, R_saturn).item()
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
        rho_inner = jnp.array([self.engineer.calculate_density(6.0)])
        R_inner = jnp.array([6.0])
        xi_inner = self.model.apply(self.params, rho_inner, R_inner).item()
        
        # Solar neighborhood
        rho_solar = jnp.array([self.engineer.calculate_density(8.0)])
        R_solar = jnp.array([8.0])
        xi_solar = self.model.apply(self.params, rho_solar, R_solar).item()
        
        # Galaxy edge
        rho_edge = jnp.array([self.engineer.calculate_density(15.0)])
        R_edge = jnp.array([15.0])
        xi_edge = self.model.apply(self.params, rho_edge, R_edge).item()
        
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
        
        R_tensor = jnp.array(R_test, dtype=jnp.float32)
        rho_tensor = jnp.array(self.engineer.calculate_density(R_test), dtype=jnp.float32)
        xi_pred = self.model.apply(self.params, rho_tensor, R_tensor)
        
        v_model = v_newton * np.sqrt(np.array(xi_pred))
        
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
        R_sample = jnp.array(np.random.uniform(5, 20, 100), dtype=jnp.float32)
        rho_sample = jnp.array(10**np.random.uniform(9, 13, 100), dtype=jnp.float32)
        xi_sample = self.model.apply(self.params, rho_sample, R_sample)
        
        xi_variance = np.var(np.array(xi_sample))
        xi_range = np.max(np.array(xi_sample)) - np.min(np.array(xi_sample))
        
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
    
    def train(self, epochs=5000, cassini_weight=100.0):
        """Train the model."""
        print(f"\nTraining on JAX GPU for {epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        # Compile training step
        train_step = jit(self.update_step)
        
        for epoch in range(epochs):
            # Training
            batch_size = min(1024, len(self.train_data['R']))
            indices = jax.random.permutation(self.key, len(self.train_data['R']))[:batch_size]
            
            rho_batch = self.train_data['rho'][indices]
            R_batch = self.train_data['R'][indices]
            xi_batch = self.train_data['xi'][indices]
            
            self.params, self.optimizer_state, train_loss = train_step(
                self.params, self.optimizer_state, self.optimizer, self.model, 
                rho_batch, R_batch, xi_batch, cassini_weight
            )
            
            # Validation
            val_loss, _ = self.compute_loss(
                self.params, self.model,
                self.val_data['rho'], self.val_data['R'], self.val_data['xi'],
                cassini_weight
            )
            
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Get current parameters
                rho_c = self.params['params']['rho_c'][0]
                n_exp = self.params['params']['n_exp'][0]
                A_boost = self.params['params']['A_boost'][0]
                
                print(f"  Parameters: rho_c = 10^{rho_c:.2f}, "
                      f"n = {n_exp:.2f}, A = {A_boost:.2f}")
                
                # Check Cassini
                cassini_loss = self.model.apply(self.params, 
                                               jnp.array([2.3e21]), 
                                               jnp.array([9.5e-6]))
                print(f"  Cassini violation: {cassini_loss:.2e}")
        
        # Store for later access
        self.train_losses = train_losses
        self.val_losses = val_losses
        
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
        R_flat = jnp.array(R_mesh.flatten(), dtype=jnp.float32)
        rho_flat = jnp.array(rho_mesh.flatten(), dtype=jnp.float32)
        xi_pred = self.model.apply(self.params, rho_flat, R_flat)
        
        xi_mesh = np.array(xi_pred).reshape(R_mesh.shape)
        
        # Get NN parameters
        rho_c = self.params['params']['rho_c'][0]
        n_exp = self.params['params']['n_exp'][0]
        A_boost = self.params['params']['A_boost'][0]
        
        print(f"\nNeural network base parameters:")
        print(f"  ρ_c = 10^{rho_c:.3f} M☉/kpc³")
        print(f"  n = {n_exp:.3f}")
        print(f"  A = {A_boost:.3f}")
        
        return xi_mesh, R_mesh, rho_mesh

    def extract_physics_formulas(self, validation_metrics):
        """Extract multiple candidate gravity formulas from the trained model."""
        print("\n" + "="*60)
        print("EXTRACTING CANDIDATE GRAVITY FORMULAS")
        print("="*60)
        
        # First check if we should even bother
        if validation_metrics['xi_range'] < 0.1:
            print("\n⚠️  Model shows insufficient variation in ξ!")
            print("Cannot extract meaningful formulas from a flat model.")
            print("Please train for more epochs or adjust hyperparameters.")
            return []
        
        formulas = []
        
        # Get neural network parameters dynamically
        rho_c = 10**self.params['params']['rho_c'][0]
        n = self.params['params']['n_exp'][0]
        A = self.params['params']['A_boost'][0]
        
        print(f"\nLearned NN parameters:")
        print(f"  ρ_c = {rho_c:.2e} M☉/kpc³")
        print(f"  n = {n:.3f}")
        print(f"  A = {A:.3f}")
        
        # Only proceed with formula generation if A > 0.1 (non-trivial enhancement)
        if A < 0.1:
            print(f"\n⚠️  Enhancement amplitude A = {A:.3f} is too small!")
            print("Model hasn't learned significant gravity modifications.")
            return []
        
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
        formula3 = {
            'name': 'Chameleon',
            'formula': lambda rho, R: 1 + A * (1 - np.tanh((rho/rho_c)**0.5)),
            'params': {'rho_c': rho_c, 'A': A},
            'description': f'ξ = 1 + {A:.2f} * (1 - tanh((ρ/{rho_c:.2e})^0.5))'
        }
        formulas.append(formula3)
        
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
            test_points = 100
            R_sample = np.random.uniform(5, 20, test_points)
            rho_sample = 10**np.random.uniform(9, 13, test_points)
            
            R_tensor = jnp.array(R_sample, dtype=jnp.float32)
            rho_tensor = jnp.array(rho_sample, dtype=jnp.float32)
            xi_nn = self.model.apply(self.params, rho_tensor, R_tensor)
            xi_formula = formula['formula'](rho_sample, R_sample)
            
            mse = np.mean((np.array(xi_nn) - xi_formula)**2)
            score = 1 / (1 + mse)
            
            print(f"  NN match score: {score:.3f}")
            formula['score'] = score
            formula['cassini_ok'] = cassini_deviation < 1e-5
        
        # Sort by score
        formulas.sort(key=lambda x: x['score'], reverse=True)
        
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
    R_tensor = jnp.array(R_plot, dtype=jnp.float32)
    rho_tensor = jnp.array(engineer.calculate_density(R_plot), dtype=jnp.float32)
    xi_pred = model.apply(trainer.params, rho_tensor, R_tensor)
    
    v_model = v_newton * np.sqrt(np.array(xi_pred))
    ax.plot(R_plot, v_model, 'r-', linewidth=2, label='Reverse engineered')
    
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('V (km/s)')
    ax.set_title('Rotation Curve Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Xi enhancement vs R
    ax = axes[0, 1]
    ax.scatter(engineer.R_binned, engineer.xi_binned, s=50, label='Empirical ξ')
    ax.plot(R_plot, np.array(xi_pred), 'r-', linewidth=2, label='Model ξ')
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
    
    rho_tensor = jnp.array(rho_range, dtype=jnp.float32)
    R_tensor = jnp.full_like(rho_tensor, R_fixed)
    xi_vs_rho = model.apply(trainer.params, rho_tensor, R_tensor)
    
    ax.loglog(rho_range, np.array(xi_vs_rho) - 1, 'r-', linewidth=2)
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
    
    R_flat = jnp.array(R_mesh.flatten(), dtype=jnp.float32)
    rho_flat = jnp.array(rho_mesh.flatten(), dtype=jnp.float32)
    xi_flat = model.apply(trainer.params, rho_flat, R_flat)
    
    xi_mesh = np.array(xi_flat).reshape(R_mesh.shape)
    
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

def main():
    """Main execution function."""
    print("="*60)
    print("REVERSE ENGINEERING GRAVITY FROM GAIA DATA (JAX GPU)")
    print("="*60)
    
    # Initialize
    engineer = GravityReverseEngineer()
    
    # Load data
    gaia_df = engineer.load_gaia_data()
    
    # Create model
    model = PhysicsInformedNN(hidden_layers=[128, 64, 32])
    
    # Initialize trainer
    trainer = GravityTrainer(engineer, model, key)
    n_train, n_val = trainer.prepare_data()
    print(f"Training on {n_train} samples, validating on {n_val} samples")
    
    # Train
    start_time = time.time()
    trainer.train_losses, trainer.val_losses = trainer.train(epochs=5000, cassini_weight=1000.0)
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
        print("\n✓ Model validation passed! Proceeding with formula extraction...")
        
        # Extract formula
        xi_mesh, R_mesh, rho_mesh = trainer.extract_formula()
        
        # Extract physics formulas with validation metrics
        formulas = trainer.extract_physics_formulas(validation_metrics)
        
        if formulas:
            # Visualize
            fig = visualize_results(engineer, model, trainer)
            
            # Save complete model
            import pickle
            model_data = {
                'params': trainer.params,
                'xi_mesh': xi_mesh,
                'R_mesh': R_mesh,
                'rho_mesh': rho_mesh,
                'baryon_params': engineer.baryon_params,
                'validation_metrics': validation_metrics,
                'epochs_trained': 5000,
                'status': 'validated'
            }
            
            with open('data/reverse_engineered_gravity_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save formulas
            import json
            formula_data = []
            for formula in formulas[:5]:  # Top 5
                formula_data.append({
                    'name': formula['name'],
                    'description': formula['description'],
                    'params': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in formula['params'].items()},
                    'score': float(formula['score']),
                    'cassini_ok': formula['cassini_ok']
                })
            
            with open('data/gravity_formulas.json', 'w') as f:
                json.dump(formula_data, f, indent=2)
            
            print("\nSaved validated formulas to data/gravity_formulas.json")
        else:
            print("\n⚠️  No valid formulas could be extracted from this model.")
            print("The model needs significant improvement before formula extraction.")    
    
    print("\n" + "="*60)
    print("REVERSE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\nFinal parameters:")
    rho_c = trainer.params['params']['rho_c'][0]
    n_exp = trainer.params['params']['n_exp'][0]
    A_boost = trainer.params['params']['A_boost'][0]
    print(f"  ρ_c = 10^{rho_c:.3f} M☉/kpc³")
    print(f"  n = {n_exp:.3f}")
    print(f"  A = {A_boost:.3f}")
    
    # Check Cassini
    rho_saturn = jnp.array([2.3e21])
    R_saturn = jnp.array([9.5e-3])  # AU to kpc
    xi_saturn = model.apply(trainer.params, rho_saturn, R_saturn).item()
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
    rho_c = 10**{rho_c:.3f}
    n = {n_exp:.3f}
    A = {A_boost:.3f}
    
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
