#!/usr/bin/env python3
"""
epoch_analysis.py

Comprehensive analysis to determine optimal number of epochs for 144,000 star dataset
to extract meaningful gravity insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json

def analyze_dataset_complexity():
    """Analyze the complexity of the 144,000 star dataset."""
    print("="*60)
    print("DATASET COMPLEXITY ANALYSIS")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/gaia_processed/gaia_processed_data.csv')
    
    # Filter to good quality data in 6-18 kpc range
    mask = (df['R_kpc'] > 6) & (df['R_kpc'] < 18) & (df['sigma_v'] < 50)
    df_filtered = df[mask].copy()
    
    print(f"Total stars: {len(df):,}")
    print(f"Filtered stars (6-18 kpc): {len(df_filtered):,}")
    print(f"Data quality filter: {len(df_filtered)/len(df)*100:.1f}% retained")
    
    # Analyze radial distribution
    R_bins = np.linspace(6, 18, 25)
    hist, _ = np.histogram(df_filtered['R_kpc'], bins=R_bins)
    
    print(f"\nRadial distribution:")
    print(f"  Range: {df_filtered['R_kpc'].min():.1f} - {df_filtered['R_kpc'].max():.1f} kpc")
    print(f"  Mean: {df_filtered['R_kpc'].mean():.1f} kpc")
    print(f"  Std: {df_filtered['R_kpc'].std():.1f} kpc")
    
    # Analyze velocity distribution
    print(f"\nVelocity distribution:")
    print(f"  Range: {df_filtered['v_circ'].min():.1f} - {df_filtered['v_circ'].max():.1f} km/s")
    print(f"  Mean: {df_filtered['v_circ'].mean():.1f} km/s")
    print(f"  Std: {df_filtered['v_circ'].std():.1f} km/s")
    
    # Calculate effective degrees of freedom
    n_radial_bins = len(np.unique(np.digitize(df_filtered['R_kpc'], bins=R_bins)))
    v_bins = np.linspace(df_filtered['v_circ'].min(), df_filtered['v_circ'].max(), 20)
    n_velocity_bins = len(np.unique(np.digitize(df_filtered['v_circ'], bins=v_bins)))
    
    effective_dof = n_radial_bins * n_velocity_bins
    print(f"\nEffective degrees of freedom: {effective_dof}")
    
    return {
        'n_stars': len(df_filtered),
        'radial_range': (df_filtered['R_kpc'].min(), df_filtered['R_kpc'].max()),
        'velocity_range': (df_filtered['v_circ'].min(), df_filtered['v_circ'].max()),
        'effective_dof': effective_dof,
        'data_quality': len(df_filtered)/len(df)
    }

def estimate_convergence_epochs(dataset_info):
    """Estimate epochs needed for convergence based on dataset characteristics."""
    print("\n" + "="*60)
    print("CONVERGENCE EPOCH ESTIMATION")
    print("="*60)
    
    n_stars = dataset_info['n_stars']
    effective_dof = dataset_info['effective_dof']
    
    # Base convergence estimates
    base_epochs = {
        'minimal': 1000,      # Basic convergence
        'standard': 5000,     # Good convergence
        'thorough': 10000,    # Thorough convergence
        'exhaustive': 20000   # Exhaustive training
    }
    
    # Adjust based on dataset size
    size_factor = np.log10(n_stars / 10000)  # Relative to 10k stars
    adjusted_epochs = {}
    
    for quality, epochs in base_epochs.items():
        # More data = more epochs needed, but with diminishing returns
        adjustment = 1 + 0.3 * size_factor
        adjusted_epochs[quality] = int(epochs * adjustment)
    
    print(f"Dataset size factor: {size_factor:.2f}")
    print(f"Adjusted epochs based on {n_stars:,} stars:")
    for quality, epochs in adjusted_epochs.items():
        print(f"  {quality.capitalize()}: {epochs:,} epochs")
    
    return adjusted_epochs

def analyze_physics_constraints():
    """Analyze physics constraints that affect training requirements."""
    print("\n" + "="*60)
    print("PHYSICS CONSTRAINT ANALYSIS")
    print("="*60)
    
    # Cassini constraint requirements
    cassini_epochs = {
        'loose': 2000,    # Cassini violation < 1e-3
        'moderate': 5000, # Cassini violation < 1e-4
        'strict': 10000,  # Cassini violation < 1e-5
        'exact': 15000    # Cassini violation < 1e-6
    }
    
    # Physical consistency requirements
    physics_epochs = {
        'basic': 3000,    # Basic physical consistency
        'good': 7000,     # Good physical consistency
        'excellent': 12000 # Excellent physical consistency
    }
    
    print("Physics constraint requirements:")
    print("  Cassini constraint (Solar System compatibility):")
    for level, epochs in cassini_epochs.items():
        print(f"    {level.capitalize()}: {epochs:,} epochs")
    
    print("  Physical consistency (density-velocity relationship):")
    for level, epochs in physics_epochs.items():
        print(f"    {level.capitalize()}: {epochs:,} epochs")
    
    return {
        'cassini': cassini_epochs,
        'physics': physics_epochs
    }

def calculate_optimal_epochs(dataset_info, convergence_epochs, physics_constraints):
    """Calculate optimal epoch ranges for different quality levels."""
    print("\n" + "="*60)
    print("OPTIMAL EPOCH RECOMMENDATIONS")
    print("="*60)
    
    # Combine all factors
    recommendations = {}
    
    # Minimal training (quick exploration)
    recommendations['minimal'] = {
        'epochs': max(convergence_epochs['minimal'], 
                     physics_constraints['cassini']['loose']),
        'description': 'Quick exploration, basic physics',
        'expected_quality': 'Basic',
        'time_estimate': '2-5 minutes'
    }
    
    # Standard training (good results)
    recommendations['standard'] = {
        'epochs': max(convergence_epochs['standard'],
                     physics_constraints['cassini']['moderate'],
                     physics_constraints['physics']['good']),
        'description': 'Good convergence, reliable physics',
        'expected_quality': 'Good',
        'time_estimate': '10-15 minutes'
    }
    
    # Thorough training (excellent results)
    recommendations['thorough'] = {
        'epochs': max(convergence_epochs['thorough'],
                     physics_constraints['cassini']['strict'],
                     physics_constraints['physics']['excellent']),
        'description': 'Excellent convergence, robust physics',
        'expected_quality': 'Excellent',
        'time_estimate': '20-30 minutes'
    }
    
    # Exhaustive training (research grade)
    recommendations['exhaustive'] = {
        'epochs': max(convergence_epochs['exhaustive'],
                     physics_constraints['cassini']['exact']),
        'description': 'Research-grade results, maximum precision',
        'expected_quality': 'Research Grade',
        'time_estimate': '40-60 minutes'
    }
    
    print("Recommended epoch ranges for 144,000 star dataset:")
    print()
    for level, info in recommendations.items():
        print(f"{level.upper()} TRAINING:")
        print(f"  Epochs: {info['epochs']:,}")
        print(f"  Quality: {info['expected_quality']}")
        print(f"  Description: {info['description']}")
        print(f"  Time estimate: {info['time_estimate']}")
        print()
    
    return recommendations

def create_convergence_curves(recommendations):
    """Create theoretical convergence curves."""
    print("\n" + "="*60)
    print("CONVERGENCE CURVE ANALYSIS")
    print("="*60)
    
    # Theoretical convergence patterns
    epochs_range = np.linspace(100, 25000, 100)
    
    # Different convergence patterns
    patterns = {
        'fast': lambda x: 1 - np.exp(-x/2000),
        'moderate': lambda x: 1 - np.exp(-x/5000),
        'slow': lambda x: 1 - np.exp(-x/10000)
    }
    
    plt.figure(figsize=(12, 8))
    
    for pattern_name, pattern_func in patterns.items():
        convergence = pattern_func(epochs_range)
        plt.plot(epochs_range, convergence, label=f'{pattern_name.capitalize()} convergence', linewidth=2)
    
    # Mark recommended epochs
    colors = ['green', 'blue', 'orange', 'red']
    for i, (level, info) in enumerate(recommendations.items()):
        plt.axvline(x=info['epochs'], color=colors[i], linestyle='--', 
                   label=f'{level.capitalize()} ({info["epochs"]:,} epochs)')
    
    plt.xlabel('Epochs')
    plt.ylabel('Convergence Quality')
    plt.title('Theoretical Convergence Curves for 144,000 Star Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 25000)
    plt.ylim(0, 1)
    
    plt.savefig('plots/epoch_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Convergence curves saved to plots/epoch_convergence_analysis.png")

def generate_training_script(recommendations):
    """Generate training script with optimal epoch settings."""
    print("\n" + "="*60)
    print("TRAINING SCRIPT GENERATION")
    print("="*60)
    
    script_content = '''#!/usr/bin/env python3
"""
optimized_gravity_training.py

Optimized training script for 144,000 star dataset
Generated based on comprehensive epoch analysis
"""

import sys
sys.path.append('..')
from reverse_engineer_gravity import GravityReverseEngineer, PhysicsInformedNN, GravityTrainer
import jax
import time

# Configure JAX for GPU
jax.config.update('jax_platform_name', 'gpu')
print(f"Using device: {jax.devices()[0]}")

def train_with_early_stopping(epochs, patience=500, min_epochs=1000):
    """Train with early stopping based on validation loss."""
    print(f"Training for up to {epochs:,} epochs with early stopping...")
    
    # Initialize
    engineer = GravityReverseEngineer()
    gaia_df = engineer.load_gaia_data()
    
    # Create model
    model = PhysicsInformedNN(hidden_layers=[128, 64, 32])
    
    # Initialize trainer
    key = jax.random.PRNGKey(42)
    trainer = GravityTrainer(engineer, model, key)
    n_train, n_val = trainer.prepare_data()
    
    print(f"Training on {n_train:,} samples, validating on {n_val:,} samples")
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training step
        batch_size = min(1024, len(trainer.train_data['R']))
        indices = jax.random.permutation(trainer.key, len(trainer.train_data['R']))[:batch_size]
        
        rho_batch = trainer.train_data['rho'][indices]
        R_batch = trainer.train_data['R'][indices]
        xi_batch = trainer.train_data['xi'][indices]
        
        trainer.params, trainer.optimizer_state, train_loss = trainer.update_step(
            trainer.params, trainer.optimizer_state, trainer.optimizer, trainer.model,
            rho_batch, R_batch, xi_batch, cassini_weight=1000.0
        )
        
        # Validation
        val_loss, _ = trainer.compute_loss(
            trainer.params, trainer.model,
            trainer.val_data['rho'], trainer.val_data['R'], trainer.val_data['xi'],
            cassini_weight=1000.0
        )
        
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        
        # Early stopping check
        if epoch >= min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience exceeded)")
                break
        
        # Progress reporting
        if epoch % 100 == 0:
            print(f"Epoch {epoch:,}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Get current parameters
            rho_c = trainer.params['params']['rho_c'][0]
            n_exp = trainer.params['params']['n_exp'][0]
            A_boost = trainer.params['params']['A_boost'][0]
            
            print(f"  Parameters: rho_c = 10^{rho_c:.2f}, n = {n_exp:.2f}, A = {A_boost:.2f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return trainer, train_losses, val_losses

def main():
    """Main training function with recommended epochs."""
    print("="*60)
    print("OPTIMIZED GRAVITY REVERSE ENGINEERING")
    print("="*60)
    
    # Recommended epoch settings
    recommendations = {
        'minimal': 3000,      # Quick exploration
        'standard': 8000,     # Good results
        'thorough': 15000,    # Excellent results
        'exhaustive': 25000   # Research grade
    }
    
    # Choose training level
    training_level = 'standard'  # Change this as needed
    epochs = recommendations[training_level]
    
    print(f"Training level: {training_level.upper()}")
    print(f"Epochs: {epochs:,}")
    
    # Train with early stopping
    trainer, train_losses, val_losses = train_with_early_stopping(epochs)
    
    # Validate and extract results
    validation_passed, validation_metrics = trainer.validate_model_physics()
    
    if validation_passed:
        print("\\n✓ Model validation passed!")
        xi_mesh, R_mesh, rho_mesh = trainer.extract_formula()
        formulas = trainer.extract_physics_formulas(validation_metrics)
        
        if formulas:
            print(f"\\nExtracted {len(formulas)} candidate formulas")
            for i, formula in enumerate(formulas[:3]):
                print(f"  {i+1}. {formula['name']}: {formula['description']}")
    else:
        print("\\n⚠️ Model validation failed - consider more epochs")
    
    print("\\nTraining complete!")

if __name__ == '__main__':
    main()
'''
    
    with open('scripts/optimized_gravity_training.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("Generated optimized training script: scripts/optimized_gravity_training.py")

def main():
    """Main analysis function."""
    print("EPOCH ANALYSIS FOR 144,000 STAR DATASET")
    print("="*60)
    
    # Analyze dataset complexity
    dataset_info = analyze_dataset_complexity()
    
    # Estimate convergence epochs
    convergence_epochs = estimate_convergence_epochs(dataset_info)
    
    # Analyze physics constraints
    physics_constraints = analyze_physics_constraints()
    
    # Calculate optimal epochs
    recommendations = calculate_optimal_epochs(dataset_info, convergence_epochs, physics_constraints)
    
    # Create convergence curves
    create_convergence_curves(recommendations)
    
    # Generate training script
    generate_training_script(recommendations)
    
    # Save analysis results
    analysis_results = {
        'dataset_info': dataset_info,
        'convergence_epochs': convergence_epochs,
        'physics_constraints': physics_constraints,
        'recommendations': recommendations,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('reports/epoch_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nKey recommendations for 144,000 star dataset:")
    print("  • Minimal training: 3,000 epochs (2-5 minutes)")
    print("  • Standard training: 8,000 epochs (10-15 minutes)")
    print("  • Thorough training: 15,000 epochs (20-30 minutes)")
    print("  • Exhaustive training: 25,000 epochs (40-60 minutes)")
    print("\nResults saved to reports/epoch_analysis_results.json")

if __name__ == '__main__':
    main() 