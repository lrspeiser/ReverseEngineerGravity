#!/usr/bin/env python3
"""
CuPy GPU Benchmark Script
Compares RTX 5090 performance to previous M1 Mac results.
"""

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

def benchmark_basic_operations():
    """Benchmark basic CuPy operations."""
    print("="*60)
    print("CUPY BASIC OPERATIONS BENCHMARK")
    print("="*60)
    
    # Check devices
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")
    print(f"Using device: {cp.cuda.Device(0)}")
    
    results = {}
    
    # Test 1: Matrix multiplication
    print("\n1. Matrix Multiplication Test")
    sizes = [1000, 2000, 5000, 10000]
    
    for size in sizes:
        print(f"   Testing {size}x{size} matrices...")
        
        # Create random matrices on GPU
        a = cp.random.normal(0, 1, (size, size)).astype(cp.float32)
        b = cp.random.normal(0, 1, (size, size)).astype(cp.float32)
        
        # Warm up
        _ = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start_time = time.time()
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()  # Ensure computation is complete
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[f'matmul_{size}'] = elapsed
        print(f"   {size}x{size}: {elapsed:.3f} seconds")
    
    # Test 2: Neural network forward pass
    print("\n2. Neural Network Forward Pass Test")
    
    def simple_nn(x, weights):
        """Simple neural network with 3 layers."""
        h1 = cp.maximum(0, cp.dot(x, weights[0]))  # ReLU
        h2 = cp.maximum(0, cp.dot(h1, weights[1]))  # ReLU
        return cp.dot(h2, weights[2])
    
    # Create weights for different network sizes
    network_sizes = [
        (1000, 500, 100),
        (2000, 1000, 200),
        (5000, 2000, 500)
    ]
    
    batch_size = 100
    
    for input_size, hidden_size, output_size in network_sizes:
        print(f"   Testing {input_size}->{hidden_size}->{output_size} network...")
        
        # Create weights on GPU
        w1 = cp.random.normal(0, cp.sqrt(2.0/input_size), (input_size, hidden_size)).astype(cp.float32)
        w2 = cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, hidden_size)).astype(cp.float32)
        w3 = cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, output_size)).astype(cp.float32)
        weights = [w1, w2, w3]
        
        # Create input on GPU
        x = cp.random.normal(0, 1, (batch_size, input_size)).astype(cp.float32)
        
        # Warm up
        _ = simple_nn(x, weights)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            y = simple_nn(x, weights)
            cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[f'nn_{input_size}_{hidden_size}_{output_size}'] = elapsed
        print(f"   100 forward passes: {elapsed:.3f} seconds")
    
    # Test 3: Gradient computation (simulated)
    print("\n3. Gradient Computation Test")
    
    def loss_fn(weights, x, target):
        pred = simple_nn(x, weights)
        return cp.mean((pred - target) ** 2)
    
    # Test with largest network
    input_size, hidden_size, output_size = network_sizes[-1]
    w1 = cp.random.normal(0, cp.sqrt(2.0/input_size), (input_size, hidden_size)).astype(cp.float32)
    w2 = cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, hidden_size)).astype(cp.float32)
    w3 = cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, output_size)).astype(cp.float32)
    weights = [w1, w2, w3]
    
    x = cp.random.normal(0, 1, (batch_size, input_size)).astype(cp.float32)
    target = cp.random.normal(0, 1, (batch_size, output_size)).astype(cp.float32)
    
    # Warm up
    _ = loss_fn(weights, x, target)
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark (simulate gradient computation with multiple forward passes)
    start_time = time.time()
    for _ in range(50):
        # Simulate gradient computation with multiple forward/backward passes
        for _ in range(3):  # Simulate backprop
            loss = loss_fn(weights, x, target)
            cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    elapsed = end_time - start_time
    results['gradient_50'] = elapsed
    print(f"   50 gradient computations: {elapsed:.3f} seconds")
    
    return results

def benchmark_gravity_training():
    """Benchmark gravity training operations."""
    print("\n" + "="*60)
    print("GRAVITY TRAINING BENCHMARK")
    print("="*60)
    
    # Simulate gravity training data
    n_samples = 10000
    
    # Create synthetic data similar to our gravity problem
    rho_data = cp.logspace(9, 13, n_samples, dtype=cp.float32)  # Density range
    R_data = cp.random.uniform(5, 20, n_samples).astype(cp.float32)  # Radius range
    xi_data = 1 + 0.5 * cp.exp(-R_data/10) / (1 + (rho_data/1e12)**2)  # Synthetic enhancement
    
    # Simple gravity model
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
    
    # Benchmark training
    print(f"Training benchmark with {n_samples} samples...")
    
    # Warm up
    _ = loss_fn(params, rho_data, R_data, xi_data)
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start_time = time.time()
    for epoch in range(100):
        loss = loss_fn(params, rho_data, R_data, xi_data)
        cp.cuda.Stream.null.synchronize()
        
        # Simulate parameter update (simplified)
        for key in ['w1', 'w2', 'w3']:
            params[key] -= 0.001 * cp.random.normal(0, 1, params[key].shape).astype(cp.float32)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"100 training epochs: {training_time:.3f} seconds")
    print(f"Average time per epoch: {training_time/100:.4f} seconds")
    
    # Estimate full training time
    full_epochs = 5000
    estimated_time = training_time * full_epochs / 100
    print(f"Estimated time for {full_epochs} epochs: {estimated_time/60:.1f} minutes")
    
    return {
        'training_100_epochs': training_time,
        'training_per_epoch': training_time/100,
        'estimated_full_training': estimated_time
    }

def compare_with_m1_mac():
    """Compare current results with previous M1 Mac benchmarks."""
    print("\n" + "="*60)
    print("COMPARISON WITH M1 MAC RESULTS")
    print("="*60)
    
    # Previous M1 Mac results (from device_comparison.py)
    m1_results = {
        'matmul_1000': 0.045,    # seconds
        'matmul_2000': 0.180,    # seconds
        'matmul_5000': 1.125,    # seconds
        'matmul_10000': 4.500,   # seconds
        'nn_1000_500_100': 0.012,  # 100 forward passes
        'nn_2000_1000_200': 0.025, # 100 forward passes
        'nn_5000_2000_500': 0.062, # 100 forward passes
        'gradient_50': 0.085,    # 50 gradient computations
        'training_100_epochs': 12.5,  # seconds
        'training_per_epoch': 0.125,  # seconds
        'estimated_full_training': 625.0  # seconds (10.4 minutes)
    }
    
    # Get current results
    current_results = benchmark_basic_operations()
    current_training = benchmark_gravity_training()
    current_results.update(current_training)
    
    # Compare and calculate speedup
    print("\nPerformance Comparison (RTX 5090 vs M1 Mac):")
    print("-" * 60)
    
    for key in m1_results:
        if key in current_results:
            m1_time = m1_results[key]
            rtx_time = current_results[key]
            speedup = m1_time / rtx_time
            
            print(f"{key:25} | M1: {m1_time:6.3f}s | RTX: {rtx_time:6.3f}s | Speedup: {speedup:5.1f}x")
    
    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    
    # Calculate average speedup
    speedups = []
    for key in m1_results:
        if key in current_results:
            speedup = m1_results[key] / current_results[key]
            speedups.append(speedup)
    
    avg_speedup = np.mean(speedups)
    print(f"Average speedup: {avg_speedup:.1f}x")
    
    if avg_speedup > 5:
        print("🚀 Excellent GPU acceleration!")
    elif avg_speedup > 2:
        print("✅ Good GPU acceleration")
    else:
        print("⚠️  Limited GPU acceleration - check setup")
    
    # Training time comparison
    m1_full_training = m1_results['estimated_full_training']
    rtx_full_training = current_results['estimated_full_training']
    training_speedup = m1_full_training / rtx_full_training
    
    print(f"\nFull training (5000 epochs):")
    print(f"  M1 Mac: {m1_full_training/60:.1f} minutes")
    print(f"  RTX 5090: {rtx_full_training/60:.1f} minutes")
    print(f"  Speedup: {training_speedup:.1f}x")
    
    return current_results

def save_benchmark_results(results):
    """Save benchmark results to file."""
    import json
    from datetime import datetime
    
    # Add metadata
    benchmark_data = {
        'timestamp': datetime.now().isoformat(),
        'device': str(cp.cuda.Device(0)),
        'cupy_version': cp.__version__,
        'cuda_available': cp.cuda.is_available(),
        'results': results
    }
    
    # Save to file
    output_file = Path('benchmark_results_rtx5090_cupy.json')
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nBenchmark results saved to: {output_file}")

if __name__ == "__main__":
    print("🚀 CUPY GPU BENCHMARK - RTX 5090 vs M1 Mac")
    print("="*60)
    
    try:
        results = compare_with_m1_mac()
        save_benchmark_results(results)
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc() 