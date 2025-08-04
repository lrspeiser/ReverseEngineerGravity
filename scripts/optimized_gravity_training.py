#!/usr/bin/env python3
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
        print("\n✓ Model validation passed!")
        xi_mesh, R_mesh, rho_mesh = trainer.extract_formula()
        formulas = trainer.extract_physics_formulas(validation_metrics)
        
        if formulas:
            print(f"\nExtracted {len(formulas)} candidate formulas")
            for i, formula in enumerate(formulas[:3]):
                print(f"  {i+1}. {formula['name']}: {formula['description']}")
    else:
        print("\n⚠️ Model validation failed - consider more epochs")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
