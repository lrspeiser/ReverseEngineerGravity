#!/usr/bin/env python3
"""
Quick CuPy GPU Test for RTX 5090
Simple test to confirm GPU usage without hanging
"""

import time
import cupy as cp
import numpy as np

def quick_gpu_test():
    """Quick test to confirm GPU is working"""
    print("=== QUICK GPU TEST ===")
    
    # Check CUDA availability
    print(f"CUDA Available: {cp.cuda.is_available()}")
    
    if not cp.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    # Get device info
    device = cp.cuda.Device(0)
    print(f"GPU Device: {str(device)}")
    print(f"GPU Memory: {device.mem_info[1] / (1024**3):.1f} GB total")
    print(f"CuPy Version: {cp.__version__}")
    
    # Quick matrix test
    print("\n=== MATRIX TEST ===")
    size = 2000
    
    # Create matrices on GPU
    a = cp.random.random((size, size))
    b = cp.random.random((size, size))
    
    # Warm up
    _ = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(5):
        c = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 5
    print(f"Matrix {size}x{size} multiplication: {avg_time*1000:.2f} ms average")
    
    # Quick gravity model test
    print("\n=== GRAVITY MODEL TEST ===")
    
    def gravity_model(r, rho_c, n_exp, A_boost):
        r_safe = cp.maximum(r, 1e-6)
        return rho_c * (r_safe / 1.0) ** (-n_exp) * (1 + A_boost * cp.exp(-r_safe / 10.0))
    
    # Generate test data
    n_points = 5000
    r = cp.linspace(1.0, 50.0, n_points)
    v_obs = 100.0 * r ** (-0.5) + cp.random.normal(0, 5.0, n_points)
    
    # Initialize parameters
    rho_c = cp.array(50.0)
    n_exp = cp.array(0.5)
    A_boost = cp.array(0.1)
    
    # Quick training test (10 epochs)
    print("Training for 10 epochs...")
    start_time = time.time()
    
    for epoch in range(10):
        # Compute loss
        v_pred = gravity_model(r, rho_c, n_exp, A_boost)
        loss = cp.mean((v_pred - v_obs) ** 2)
        
        # Simple gradient descent
        eps = 1e-6
        learning_rate = 0.01
        
        # Gradients (finite differences)
        loss_plus = cp.mean((gravity_model(r, rho_c + eps, n_exp, A_boost) - v_obs) ** 2)
        grad_rho_c = (loss_plus - loss) / eps
        
        loss_plus = cp.mean((gravity_model(r, rho_c, n_exp + eps, A_boost) - v_obs) ** 2)
        grad_n_exp = (loss_plus - loss) / eps
        
        loss_plus = cp.mean((gravity_model(r, rho_c, n_exp, A_boost + eps) - v_obs) ** 2)
        grad_A_boost = (loss_plus - loss) / eps
        
        # Update parameters
        rho_c -= learning_rate * grad_rho_c
        n_exp -= learning_rate * grad_n_exp
        A_boost -= learning_rate * grad_A_boost
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}, Loss: {float(loss):.6f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Time per epoch: {training_time/10*1000:.2f} ms")
    print(f"Estimated 5000 epochs: {training_time/10*5000/60:.1f} minutes")
    
    print(f"\nFinal parameters:")
    print(f"  rho_c: {float(rho_c):.4f}")
    print(f"  n_exp: {float(n_exp):.4f}")
    print(f"  A_boost: {float(A_boost):.4f}")
    
    return True

if __name__ == "__main__":
    print("Quick CuPy GPU Test for RTX 5090")
    print("=" * 40)
    
    success = quick_gpu_test()
    
    if success:
        print("\n" + "=" * 40)
        print("GPU TEST COMPLETED SUCCESSFULLY!")
        print("RTX 5090 is being used for computations.")
    else:
        print("\nGPU test failed!") 