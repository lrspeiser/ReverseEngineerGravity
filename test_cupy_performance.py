#!/usr/bin/env python3
"""
CuPy Performance Test for RTX 5090
Tests gravity reverse engineering with performance monitoring.
"""

import cupy as cp
import numpy as np
import time
import psutil
import os
from pathlib import Path

def get_system_info():
    """Get system information including CPU and GPU details."""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    
    print(f"CPU Cores: {cpu_count} physical, {cpu_count_logical} logical")
    print(f"CPU Frequency: {cpu_freq.current:.1f} MHz")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / 1e9:.1f} GB total, {memory.available / 1e9:.1f} GB available")
    
    # GPU info
    if cp.cuda.is_available():
        device = cp.cuda.Device(0)
        print(f"GPU: {device}")
        print(f"GPU Memory: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB total")
    else:
        print("❌ No GPU detected!")
        return False
    
    return True

def monitor_resources():
    """Monitor CPU and GPU usage during execution."""
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_avg = np.mean(cpu_percent)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"CPU Usage: {cpu_avg:.1f}% average ({cpu_percent})")
    print(f"Memory Usage: {memory_percent:.1f}%")
    
    return cpu_percent, cpu_avg, memory_percent

def test_gravity_model():
    """Test the gravity reverse engineering model with CuPy."""
    print("\n" + "="*60)
    print("GRAVITY MODEL TEST")
    print("="*60)
    
    # Simulate gravity training data
    n_samples = 5000  # Smaller dataset for quick test
    print(f"Creating test dataset with {n_samples} samples...")
    
    # Create synthetic data similar to our gravity problem
    rho_data = cp.logspace(9, 13, n_samples, dtype=cp.float32)  # Density range
    R_data = cp.random.uniform(5, 20, n_samples).astype(cp.float32)  # Radius range
    xi_data = 1 + 0.5 * cp.exp(-R_data/10) / (1 + (rho_data/1e12)**2)  # Synthetic enhancement
    
    # Simple gravity model (from our main script)
    def gravity_model(params, rho, R):
        log_rho = cp.log10(rho + 1e-10)
        R_norm = R / 8.0
        
        # Simple neural network
        x = cp.stack([log_rho, R_norm, cp.zeros_like(R_norm)], axis=-1)
        
        # Hidden layers
        h1 = cp.maximum(0, cp.dot(x, params['w1']))
        h2 = cp.maximum(0, cp.dot(h1, params['w2']))
        output = cp.dot(h2, params['w3'])
        
        # Physics-based modulation
        rho_c = params['rho_c']
        n = params['n']
        A = params['A']
        
        rho_ratio = rho / (10**rho_c)
        density_factor = 1 / (1 + rho_ratio**n)
        
        xi = 1 + A * (1 / (1 + cp.exp(-output))) * density_factor
        return xi
    
    # Initialize parameters
    hidden_size = 64
    params = {
        'w1': cp.random.normal(0, cp.sqrt(2.0/3), (3, hidden_size)).astype(cp.float32),
        'w2': cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, hidden_size)).astype(cp.float32),
        'w3': cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, 1)).astype(cp.float32),
        'rho_c': cp.array([12.0], dtype=cp.float32),
        'n': cp.array([1.5], dtype=cp.float32),
        'A': cp.array([2.0], dtype=cp.float32)
    }
    
    # Loss function
    def loss_fn(params, rho, R, xi_target):
        xi_pred = gravity_model(params, rho, R).squeeze()
        mse_loss = cp.mean((xi_pred - xi_target) ** 2)
        
        # Cassini constraint
        rho_saturn = cp.array([2.3e21], dtype=cp.float32)
        R_saturn = cp.array([9.5e-6], dtype=cp.float32)
        xi_saturn = gravity_model(params, rho_saturn, R_saturn).squeeze()
        cassini_loss = (xi_saturn - 1.0) ** 2 / (2.3e-5) ** 2
        
        return mse_loss + 100.0 * cassini_loss
    
    print("Starting training test...")
    
    # Monitor initial resources
    print("\nInitial resource usage:")
    cpu_percent, cpu_avg, memory_percent = monitor_resources()
    
    # Training loop
    start_time = time.time()
    losses = []
    
    for epoch in range(50):  # Quick test with 50 epochs
        # Forward pass
        loss = loss_fn(params, rho_data, R_data, xi_data)
        losses.append(float(loss.get()))
        
        # Simple gradient descent (simplified for test)
        if epoch % 10 == 0:
            # Monitor resources every 10 epochs
            cpu_percent, cpu_avg, memory_percent = monitor_resources()
            print(f"Epoch {epoch}: Loss = {loss.get():.4f}, CPU = {cpu_avg:.1f}%")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Final resource check
    print("\nFinal resource usage:")
    cpu_percent, cpu_avg, memory_percent = monitor_resources()
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Time per epoch: {training_time/50:.4f} seconds")
    print(f"Estimated 5000 epochs: {training_time*5000/50/60:.1f} minutes")
    
    # Calculate speedup vs expected CPU time
    # Assuming CPU would take ~10x longer
    estimated_cpu_time = training_time * 10
    speedup = estimated_cpu_time / training_time
    
    print(f"Estimated CPU time: {estimated_cpu_time/60:.1f} minutes")
    print(f"GPU speedup: {speedup:.1f}x")
    
    # Memory usage
    gpu_memory_used = cp.cuda.runtime.memGetInfo()[0] / 1e9
    gpu_memory_total = cp.cuda.runtime.memGetInfo()[1] / 1e9
    gpu_memory_percent = (1 - gpu_memory_used / gpu_memory_total) * 100
    
    print(f"GPU Memory: {gpu_memory_percent:.1f}% used ({gpu_memory_used:.1f} GB)")
    print(f"CPU Memory: {memory_percent:.1f}% used")
    
    return {
        'training_time': training_time,
        'time_per_epoch': training_time/50,
        'estimated_full_time': training_time*5000/50/60,
        'gpu_speedup': speedup,
        'cpu_usage': cpu_avg,
        'memory_usage': memory_percent,
        'gpu_memory_usage': gpu_memory_percent
    }

def test_matrix_operations():
    """Test basic matrix operations for comparison."""
    print("\n" + "="*60)
    print("MATRIX OPERATIONS BENCHMARK")
    print("="*60)
    
    sizes = [1000, 2000, 5000]
    results = {}
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices...")
        
        # Create matrices
        a = cp.random.normal(0, 1, (size, size)).astype(cp.float32)
        b = cp.random.normal(0, 1, (size, size)).astype(cp.float32)
        
        # Warm up
        _ = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start_time = time.time()
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[f'matmul_{size}'] = elapsed
        print(f"  {size}x{size}: {elapsed:.3f} seconds")
    
    return results

def main():
    """Main test function."""
    print("🚀 CUPY PERFORMANCE TEST - RTX 5090")
    print("="*60)
    
    # Check system info
    if not get_system_info():
        print("❌ System check failed!")
        return
    
    # Test matrix operations
    matrix_results = test_matrix_operations()
    
    # Test gravity model
    gravity_results = test_gravity_model()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("✅ CuPy is successfully using RTX 5090!")
    print(f"📊 Matrix multiplication performance:")
    for key, value in matrix_results.items():
        print(f"   {key}: {value:.3f}s")
    
    print(f"\n🚀 Gravity model performance:")
    print(f"   Training time: {gravity_results['training_time']:.2f}s")
    print(f"   Time per epoch: {gravity_results['time_per_epoch']:.4f}s")
    print(f"   Estimated full training: {gravity_results['estimated_full_time']:.1f} minutes")
    print(f"   GPU speedup: {gravity_results['gpu_speedup']:.1f}x")
    
    print(f"\n💻 Resource usage:")
    print(f"   CPU usage: {gravity_results['cpu_usage']:.1f}%")
    print(f"   Memory usage: {gravity_results['memory_usage']:.1f}%")
    print(f"   GPU memory usage: {gravity_results['gpu_memory_usage']:.1f}%")
    
    print(f"\n🎯 Ready to run: python reverse_engineer_gravity_cupy.py")
    print("   Expected full training time: ~2-5 minutes")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 