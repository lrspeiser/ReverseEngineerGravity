#!/usr/bin/env python3
"""
CuPy GPU Performance Test for RTX 5090
Tests gravity reverse engineering model performance and system resource usage
"""

import time
import cupy as cp
import numpy as np
import psutil
import os

def get_system_info():
    """Get detailed system information"""
    print("=== SYSTEM INFORMATION ===")
    
    # CPU Information
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"CPU Cores: {cpu_count_physical} physical, {cpu_count_logical} logical")
    print(f"CPU Frequency: {cpu_freq.current:.1f} MHz")
    print(f"RAM: {ram_gb:.1f} GB")
    
    # GPU Information
    if cp.cuda.is_available():
        device = cp.cuda.Device(0)
        print(f"GPU Device: {str(device)}")
        print(f"GPU Memory: {device.mem_info[1] / (1024**3):.1f} GB total")
        print(f"CuPy Version: {cp.__version__}")
        print(f"CUDA Available: {cp.cuda.is_available()}")
    else:
        print("CUDA not available!")
        return False
    
    return True

def monitor_resources():
    """Monitor current CPU and memory usage"""
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_avg = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"CPU Usage: {cpu_avg:.1f}% average")
    print(f"CPU per core: {[f'{p:.1f}%' for p in cpu_percent]}")
    print(f"Memory Usage: {memory_percent:.1f}%")
    
    return cpu_avg, cpu_percent, memory_percent

def test_matrix_operations():
    """Test basic matrix operations for GPU speed"""
    print("\n=== MATRIX OPERATIONS BENCHMARK ===")
    
    sizes = [1000, 2000, 4000]
    results = {}
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrix multiplication...")
        
        # Create test matrices
        a = cp.random.random((size, size))
        b = cp.random.random((size, size))
        
        # Warm up
        _ = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        results[size] = avg_time
        print(f"  Average time: {avg_time*1000:.2f} ms")
    
    return results

def test_gravity_model():
    """Test simplified gravity reverse engineering model"""
    print("\n=== GRAVITY MODEL PERFORMANCE TEST ===")
    
    # Simplified gravity model (similar to main code)
    def gravity_model(r, rho_c, n_exp, A_boost):
        """Simplified gravity model for testing"""
        r_safe = cp.maximum(r, 1e-6)
        return rho_c * (r_safe / 1.0) ** (-n_exp) * (1 + A_boost * cp.exp(-r_safe / 10.0))
    
    def loss_fn(r, v_obs, rho_c, n_exp, A_boost):
        """Simplified loss function"""
        v_pred = gravity_model(r, rho_c, n_exp, A_boost)
        mse_loss = cp.mean((v_pred - v_obs) ** 2)
        return mse_loss
    
    # Generate test data
    print("Generating test data...")
    n_points = 10000
    r = cp.linspace(1.0, 50.0, n_points)
    v_obs = 100.0 * r ** (-0.5) + cp.random.normal(0, 5.0, n_points)
    
    # Initialize parameters
    rho_c = cp.array(50.0)
    n_exp = cp.array(0.5)
    A_boost = cp.array(0.1)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 50
    
    print(f"Training for {epochs} epochs...")
    print("Monitoring resources every 10 epochs...")
    
    start_time = time.time()
    losses = []
    
    for epoch in range(epochs):
        # Compute loss
        loss = loss_fn(r, v_obs, rho_c, n_exp, A_boost)
        losses.append(float(loss))
        
        # Simple gradient descent (finite differences)
        eps = 1e-6
        
        # Gradient for rho_c
        loss_plus = loss_fn(r, v_obs, rho_c + eps, n_exp, A_boost)
        grad_rho_c = (loss_plus - loss) / eps
        
        # Gradient for n_exp
        loss_plus = loss_fn(r, v_obs, rho_c, n_exp + eps, A_boost)
        grad_n_exp = (loss_plus - loss) / eps
        
        # Gradient for A_boost
        loss_plus = loss_fn(r, v_obs, rho_c, n_exp, A_boost + eps)
        grad_A_boost = (loss_plus - loss) / eps
        
        # Update parameters
        rho_c -= learning_rate * grad_rho_c
        n_exp -= learning_rate * grad_n_exp
        A_boost -= learning_rate * grad_A_boost
        
        # Monitor resources every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {float(loss):.6f}")
            cpu_avg, cpu_percent, mem_percent = monitor_resources()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Average time per epoch: {training_time/epochs*1000:.2f} ms")
    print(f"Estimated time for 5000 epochs: {training_time/epochs*5000/60:.1f} minutes")
    
    # Final parameter values
    print(f"Final parameters:")
    print(f"  rho_c: {float(rho_c):.4f}")
    print(f"  n_exp: {float(n_exp):.4f}")
    print(f"  A_boost: {float(A_boost):.4f}")
    
    return training_time, losses

def main():
    """Main performance test"""
    print("CuPy GPU Performance Test for RTX 5090")
    print("=" * 50)
    
    # Check system info
    if not get_system_info():
        print("ERROR: CUDA not available. Cannot proceed with GPU test.")
        return
    
    # Initial resource check
    print("\n=== INITIAL RESOURCE USAGE ===")
    monitor_resources()
    
    # Test matrix operations
    matrix_results = test_matrix_operations()
    
    # Test gravity model
    training_time, losses = test_gravity_model()
    
    # Final resource check
    print("\n=== FINAL RESOURCE USAGE ===")
    monitor_resources()
    
    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Matrix multiplication performance:")
    for size, time_ms in matrix_results.items():
        print(f"  {size}x{size}: {time_ms*1000:.2f} ms")
    
    print(f"\nGravity model training:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Time per epoch: {training_time/50*1000:.2f} ms")
    print(f"  Estimated 5000 epochs: {training_time/50*5000/60:.1f} minutes")
    
    # Save results
    results = {
        'matrix_results': matrix_results,
        'training_time': training_time,
        'time_per_epoch': training_time/50,
        'estimated_full_training': training_time/50*5000,
        'gpu_device': str(cp.cuda.Device(0)),
        'cupy_version': cp.__version__
    }
    
    print(f"\nResults saved to: performance_test_results_cupy.json")
    print("GPU test completed successfully!")

if __name__ == "__main__":
    main() 