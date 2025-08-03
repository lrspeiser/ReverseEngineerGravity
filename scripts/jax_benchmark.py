#!/usr/bin/env python3
"""
JAX GPU Benchmark Script
Compares RTX 5090 performance to previous M1 Mac results.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def benchmark_basic_operations():
    """Benchmark basic JAX operations."""
    print("="*60)
    print("JAX BASIC OPERATIONS BENCHMARK")
    print("="*60)
    
    # Check devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        print(f"✅ GPU found: {gpu_devices[0]}")
        device = gpu_devices[0]
    else:
        print("⚠️  No GPU found, using CPU")
        device = devices[0]
    
    results = {}
    
    # Test 1: Matrix multiplication
    print("\n1. Matrix Multiplication Test")
    sizes = [1000, 2000, 5000, 10000]
    
    for size in sizes:
        print(f"   Testing {size}x{size} matrices...")
        
        # Create random matrices
        a = jax.random.normal(jax.random.PRNGKey(0), (size, size))
        b = jax.random.normal(jax.random.PRNGKey(1), (size, size))
        
        # Warm up
        _ = jnp.dot(a, b)
        
        # Benchmark
        start_time = time.time()
        c = jnp.dot(a, b)
        c.block_until_ready()  # Ensure computation is complete
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[f'matmul_{size}'] = elapsed
        print(f"   {size}x{size}: {elapsed:.3f} seconds")
    
    # Test 2: Neural network forward pass
    print("\n2. Neural Network Forward Pass Test")
    
    def simple_nn(x, weights):
        """Simple neural network with 3 layers."""
        h1 = jax.nn.relu(jnp.dot(x, weights[0]))
        h2 = jax.nn.relu(jnp.dot(h1, weights[1]))
        return jnp.dot(h2, weights[2])
    
    # Create weights for different network sizes
    network_sizes = [
        (1000, 500, 100),
        (2000, 1000, 200),
        (5000, 2000, 500)
    ]
    
    batch_size = 100
    
    for input_size, hidden_size, output_size in network_sizes:
        print(f"   Testing {input_size}->{hidden_size}->{output_size} network...")
        
        # Create weights
        w1 = jax.random.normal(jax.random.PRNGKey(2), (input_size, hidden_size))
        w2 = jax.random.normal(jax.random.PRNGKey(3), (hidden_size, hidden_size))
        w3 = jax.random.normal(jax.random.PRNGKey(4), (hidden_size, output_size))
        weights = [w1, w2, w3]
        
        # Create input
        x = jax.random.normal(jax.random.PRNGKey(5), (batch_size, input_size))
        
        # Warm up
        _ = simple_nn(x, weights)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            y = simple_nn(x, weights)
            y.block_until_ready()
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[f'nn_{input_size}_{hidden_size}_{output_size}'] = elapsed
        print(f"   100 forward passes: {elapsed:.3f} seconds")
    
    # Test 3: Gradient computation
    print("\n3. Gradient Computation Test")
    
    def loss_fn(weights, x, target):
        pred = simple_nn(x, weights)
        return jnp.mean((pred - target) ** 2)
    
    # Test with largest network
    input_size, hidden_size, output_size = network_sizes[-1]
    w1 = jax.random.normal(jax.random.PRNGKey(6), (input_size, hidden_size))
    w2 = jax.random.normal(jax.random.PRNGKey(7), (hidden_size, hidden_size))
    w3 = jax.random.normal(jax.random.PRNGKey(8), (hidden_size, output_size))
    weights = [w1, w2, w3]
    
    x = jax.random.normal(jax.random.PRNGKey(9), (batch_size, input_size))
    target = jax.random.normal(jax.random.PRNGKey(10), (batch_size, output_size))
    
    # Compile gradient function
    grad_fn = jax.grad(loss_fn, argnums=0)
    
    # Warm up
    _ = grad_fn(weights, x, target)
    
    # Benchmark
    start_time = time.time()
    for _ in range(50):
        grads = grad_fn(weights, x, target)
        # Ensure all gradients are computed
        for grad in grads:
            grad.block_until_ready()
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
    n_features = 3
    
    # Create synthetic data similar to our gravity problem
    rho_data = jnp.logspace(9, 13, n_samples)  # Density range
    R_data = jax.random.uniform(jax.random.PRNGKey(11), (n_samples,), minval=5, maxval=20)  # Radius range
    xi_data = 1 + 0.5 * jnp.exp(-R_data/10) / (1 + (rho_data/1e12)**2)  # Synthetic enhancement
    
    # Simple gravity model
    def gravity_model(params, rho, R):
        log_rho = jnp.log10(rho + 1e-10)
        R_norm = R / 8.0
        
        # Simple neural network
        x = jnp.stack([log_rho, R_norm, jnp.zeros_like(R_norm)], axis=-1)
        
        # Hidden layers
        h1 = jax.nn.relu(jnp.dot(x, params['w1']))
        h2 = jax.nn.relu(jnp.dot(h1, params['w2']))
        output = jnp.dot(h2, params['w3'])
        
        # Physics-based modulation
        rho_c = params['rho_c']
        n = params['n']
        A = params['A']
        
        rho_ratio = rho / (10**rho_c)
        density_factor = 1 / (1 + rho_ratio**n)
        
        xi = 1 + A * jax.nn.sigmoid(output) * density_factor
        return xi
    
    # Initialize parameters
    hidden_size = 64
    params = {
        'w1': jax.random.normal(jax.random.PRNGKey(12), (3, hidden_size)),
        'w2': jax.random.normal(jax.random.PRNGKey(13), (hidden_size, hidden_size)),
        'w3': jax.random.normal(jax.random.PRNGKey(14), (hidden_size, 1)),
        'rho_c': jnp.array([12.0]),
        'n': jnp.array([1.5]),
        'A': jnp.array([2.0])
    }
    
    # Loss function
    def loss_fn(params, rho, R, xi_target):
        xi_pred = gravity_model(params, rho, R).squeeze()
        mse_loss = jnp.mean((xi_pred - xi_target) ** 2)
        
        # Cassini constraint
        rho_saturn = jnp.array([2.3e21])
        R_saturn = jnp.array([9.5e-6])
        xi_saturn = gravity_model(params, rho_saturn, R_saturn).squeeze()
        cassini_loss = (xi_saturn - 1.0) ** 2 / (2.3e-5) ** 2
        
        return mse_loss + 100.0 * cassini_loss
    
    # Compile training step
    grad_fn = jax.grad(loss_fn)
    
    # Benchmark training
    print(f"Training benchmark with {n_samples} samples...")
    
    # Warm up
    _ = grad_fn(params, rho_data, R_data, xi_data)
    
    # Benchmark
    start_time = time.time()
    for epoch in range(100):
        grads = grad_fn(params, rho_data, R_data, xi_data)
        # Simulate parameter update
        for key in params:
            if isinstance(params[key], jnp.ndarray):
                params[key] = params[key] - 0.001 * grads[key]
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
        'device': str(jax.devices()[0]),
        'jax_version': jax.__version__,
        'results': results
    }
    
    # Save to file
    output_file = Path('benchmark_results_rtx5090.json')
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nBenchmark results saved to: {output_file}")

if __name__ == "__main__":
    print("🚀 JAX GPU BENCHMARK - RTX 5090 vs M1 Mac")
    print("="*60)
    
    try:
        results = compare_with_m1_mac()
        save_benchmark_results(results)
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc() 