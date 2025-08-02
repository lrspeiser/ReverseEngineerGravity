#!/usr/bin/env python3
"""
test_reverse_engineering.py

Test script to verify that the reverse engineering code works correctly
with our organized Gaia data structure.
"""

import pandas as pd
import numpy as np
from reverse_engineer_gravity import GravityReverseEngineer, PhysicsInformedNN, GravityTrainer

def test_data_loading():
    """Test that the data loading works correctly."""
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    # Test data loading
    engineer = GravityReverseEngineer()
    df = engineer.load_gaia_data()
    
    print(f"✅ Successfully loaded {len(df)} stars")
    print(f"✅ Distance range: {engineer.R_data.min():.2f} - {engineer.R_data.max():.2f} kpc")
    print(f"✅ Velocity range: {engineer.v_data.min():.2f} - {engineer.v_data.max():.2f} km/s")
    print(f"✅ Error range: {engineer.sigma_v.min():.2f} - {engineer.sigma_v.max():.2f} km/s")
    
    return engineer

def test_empirical_xi():
    """Test the empirical xi calculation."""
    print("\n" + "="*60)
    print("TESTING EMPIRICAL XI CALCULATION")
    print("="*60)
    
    engineer = GravityReverseEngineer()
    df = engineer.load_gaia_data()
    R_binned, xi_binned = engineer.derive_empirical_xi()
    
    print(f"✅ Successfully calculated empirical xi for {len(R_binned)} bins")
    print(f"✅ Xi range: {xi_binned.min():.4f} - {xi_binned.max():.4f}")
    print(f"✅ R range: {R_binned.min():.2f} - {R_binned.max():.2f} kpc")
    
    return engineer, R_binned, xi_binned

def test_model_creation():
    """Test that the neural network model can be created."""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION")
    print("="*60)
    
    # Create model
    model = PhysicsInformedNN(hidden_layers=[64, 32])
    print(f"✅ Successfully created neural network model")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,} total")
    
    return model

def test_training_setup():
    """Test that the training setup works."""
    print("\n" + "="*60)
    print("TESTING TRAINING SETUP")
    print("="*60)
    
    engineer = GravityReverseEngineer()
    df = engineer.load_gaia_data()
    model = PhysicsInformedNN(hidden_layers=[64, 32])
    trainer = GravityTrainer(engineer, model)
    
    # Prepare data
    train_dataset, val_dataset = trainer.prepare_data()
    print(f"✅ Successfully prepared training data")
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    return trainer

def test_short_training():
    """Test a short training run."""
    print("\n" + "="*60)
    print("TESTING SHORT TRAINING RUN")
    print("="*60)
    
    engineer = GravityReverseEngineer()
    df = engineer.load_gaia_data()
    model = PhysicsInformedNN(hidden_layers=[64, 32])
    trainer = GravityTrainer(engineer, model)
    trainer.prepare_data()
    
    # Run short training
    print("Running 10 epochs of training...")
    train_losses, val_losses = trainer.train(epochs=10, cassini_weight=100.0)
    
    print(f"✅ Training completed successfully")
    print(f"✅ Final training loss: {train_losses[-1]:.4f}")
    print(f"✅ Final validation loss: {val_losses[-1]:.4f}")
    
    return trainer

def main():
    """Run all tests."""
    print("🧪 REVERSE ENGINEERING CODE TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Data loading
        engineer = test_data_loading()
        
        # Test 2: Empirical xi calculation
        engineer, R_binned, xi_binned = test_empirical_xi()
        
        # Test 3: Model creation
        model = test_model_creation()
        
        # Test 4: Training setup
        trainer = test_training_setup()
        
        # Test 5: Short training
        trainer = test_short_training()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("✅ The reverse engineering code is ready to use with your organized Gaia data!")
        print("✅ You can now run: python reverse_engineer_gravity.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 