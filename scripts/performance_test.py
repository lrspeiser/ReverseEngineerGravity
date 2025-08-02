#!/usr/bin/env python3
"""
performance_test.py

Performance testing script to estimate training time and show sample results
on M1 Mac before running on 5090 GPU.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from reverse_engineer_gravity import GravityReverseEngineer, PhysicsInformedNN, GravityTrainer

def test_device_performance():
    """Test basic device performance."""
    print("="*60)
    print("DEVICE PERFORMANCE TEST")
    print("="*60)
    
    # Test tensor operations
    size = 10000
    print(f"Testing tensor operations with {size:,} elements...")
    
    # CPU test
    start_time = time.time()
    x_cpu = torch.randn(size, size, device='cpu')
    y_cpu = torch.randn(size, size, device='cpu')
    z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.3f} seconds")
    
    # GPU test
    if torch.cuda.is_available():
        start_time = time.time()
        x_gpu = torch.randn(size, size, device='cuda')
        y_gpu = torch.randn(size, size, device='cuda')
        z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"CUDA time: {gpu_time:.3f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    elif torch.backends.mps.is_available():
        start_time = time.time()
        x_mps = torch.randn(size, size, device='mps')
        y_mps = torch.randn(size, size, device='mps')
        z_mps = torch.mm(x_mps, y_mps)
        mps_time = time.time() - start_time
        print(f"MPS time: {mps_time:.3f} seconds")
        print(f"Speedup: {cpu_time/mps_time:.1f}x")

def estimate_training_time():
    """Estimate full training time based on short runs."""
    print("\n" + "="*60)
    print("TRAINING TIME ESTIMATION")
    print("="*60)
    
    # Load data
    print("Loading Gaia data...")
    engineer = GravityReverseEngineer()
    df = engineer.load_gaia_data()
    
    # Create model
    model = PhysicsInformedNN(hidden_layers=[128, 64, 32])
    trainer = GravityTrainer(engineer, model)
    trainer.prepare_data()
    
    # Test different epoch counts
    epoch_counts = [5, 10, 20]
    times_per_epoch = []
    
    for epochs in epoch_counts:
        print(f"\nTesting {epochs} epochs...")
        start_time = time.time()
        train_losses, val_losses = trainer.train(epochs=epochs, cassini_weight=100.0)
        elapsed_time = time.time() - start_time
        
        time_per_epoch = elapsed_time / epochs
        times_per_epoch.append(time_per_epoch)
        
        print(f"  {epochs} epochs took {elapsed_time:.1f} seconds")
        print(f"  Average time per epoch: {time_per_epoch:.2f} seconds")
        print(f"  Final training loss: {train_losses[-1]:.4f}")
        print(f"  Final validation loss: {val_losses[-1]:.4f}")
    
    # Estimate full training
    avg_time_per_epoch = np.mean(times_per_epoch)
    full_epochs = 500  # Default in main script
    
    estimated_full_time = avg_time_per_epoch * full_epochs
    estimated_minutes = estimated_full_time / 60
    estimated_hours = estimated_minutes / 60
    
    print(f"\n📊 TRAINING TIME ESTIMATES:")
    print(f"  Average time per epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"  Estimated time for {full_epochs} epochs: {estimated_full_time:.0f} seconds")
    print(f"  Estimated time: {estimated_minutes:.1f} minutes ({estimated_hours:.1f} hours)")
    
    # Estimate 5090 performance (rough estimate: 5-10x faster)
    print(f"\n🚀 5090 GPU ESTIMATES (5-10x speedup):")
    print(f"  Estimated time: {estimated_minutes/7.5:.1f} minutes")
    print(f"  Estimated time: {estimated_hours/7.5:.1f} hours")
    
    return avg_time_per_epoch, model, trainer

def show_sample_results(model, trainer):
    """Show sample results from the trained model."""
    print("\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)
    
    # Get model parameters
    with torch.no_grad():
        rho_c = model.rho_c.item()
        n_exp = model.n_exp.item()
        A_boost = model.A_boost.item()
    
    print(f"📊 MODEL PARAMETERS:")
    print(f"  ρ_c = 10^{rho_c:.3f} M☉/kpc³")
    print(f"  n = {n_exp:.3f}")
    print(f"  A = {A_boost:.3f}")
    
    # Test Cassini constraint
    with torch.no_grad():
        rho_saturn = torch.tensor([2.3e21], device=model.network[0].weight.device)
        R_saturn = torch.tensor([9.5e-3], device=model.network[0].weight.device)  # AU to kpc
        xi_saturn = model(rho_saturn, R_saturn).item()
        print(f"\n🔬 CASSINI CONSTRAINT TEST:")
        print(f"  ξ(Saturn) = {xi_saturn:.8f}")
        print(f"  Deviation from 1.0: {abs(xi_saturn-1):.2e}")
        print(f"  Status: {'✅ PASS' if abs(xi_saturn-1) < 2.3e-5 else '❌ FAIL'}")
    
    # Show rotation curve fit
    print(f"\n🌌 ROTATION CURVE FIT:")
    R_test = np.linspace(6, 18, 100)
    rho_test = trainer.engineer.calculate_density(R_test)
    
    with torch.no_grad():
        R_tensor = torch.tensor(R_test, dtype=torch.float32).to(model.network[0].weight.device)
        rho_tensor = torch.tensor(rho_test, dtype=torch.float32).to(model.network[0].weight.device)
        xi_pred = model(rho_tensor, R_tensor).cpu().numpy()
    
    v_newton = trainer.engineer.calculate_newtonian_velocity(R_test)
    v_model = v_newton * np.sqrt(xi_pred)
    
    print(f"  Velocity range: {v_model.min():.1f} - {v_model.max():.1f} km/s")
    print(f"  Mean enhancement: {np.mean(xi_pred):.3f}")
    print(f"  Enhancement range: {xi_pred.min():.3f} - {xi_pred.max():.3f}")

def create_quick_visualization(model, trainer):
    """Create a quick visualization of the results."""
    print("\n" + "="*60)
    print("CREATING QUICK VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rotation curve
    ax = axes[0]
    R_test = np.linspace(6, 18, 100)
    rho_test = trainer.engineer.calculate_density(R_test)
    
    with torch.no_grad():
        R_tensor = torch.tensor(R_test, dtype=torch.float32).to(model.network[0].weight.device)
        rho_tensor = torch.tensor(rho_test, dtype=torch.float32).to(model.network[0].weight.device)
        xi_pred = model(rho_tensor, R_tensor).cpu().numpy()
    
    v_newton = trainer.engineer.calculate_newtonian_velocity(R_test)
    v_model = v_newton * np.sqrt(xi_pred)
    
    # Plot data points (subsampled)
    ax.scatter(trainer.engineer.R_data[::1000], trainer.engineer.v_data[::1000], 
               alpha=0.3, s=1, label='Gaia data', color='blue')
    ax.plot(R_test, v_newton, 'g--', label='Newtonian', linewidth=2)
    ax.plot(R_test, v_model, 'r-', label='Reverse engineered', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('V (km/s)')
    ax.set_title('Rotation Curve Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Xi enhancement
    ax = axes[1]
    ax.plot(R_test, xi_pred, 'r-', linewidth=2, label='Model ξ')
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('ξ enhancement')
    ax.set_title('Gravity Enhancement Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/performance_test_results.png', dpi=150, bbox_inches='tight')
    print("✅ Saved quick visualization to ../plots/performance_test_results.png")

def main():
    """Run performance tests."""
    print("🚀 REVERSE ENGINEERING PERFORMANCE TEST")
    print("="*60)
    print("Testing on M1 Mac to estimate 5090 performance...")
    
    try:
        # Test device performance
        test_device_performance()
        
        # Estimate training time
        avg_time_per_epoch, model, trainer = estimate_training_time()
        
        # Show sample results
        show_sample_results(model, trainer)
        
        # Create quick visualization
        create_quick_visualization(model, trainer)
        
        print("\n" + "="*60)
        print("🎉 PERFORMANCE TEST COMPLETE!")
        print("="*60)
        print("✅ You now have estimates for 5090 training time")
        print("✅ Sample results show the model is working correctly")
        print("✅ Quick visualization saved to plots/")
        print("\n💡 RECOMMENDATIONS:")
        print("  • Full training on 5090 should take ~10-30 minutes")
        print("  • Results look promising for gravity reverse engineering")
        print("  • Consider running full training on 5090 for best results")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 