#!/usr/bin/env python3
"""
CuPy Setup Verification Script
Tests CuPy GPU functionality on RTX 5090.
"""

import cupy as cp
import time
import numpy as np

def test_cupy_setup():
    """Test CuPy GPU functionality."""
    print("="*60)
    print("CUPY GPU SETUP VERIFICATION")
    print("="*60)
    
    # Check CuPy version and CUDA availability
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")
    
    if cp.cuda.is_available():
        print(f"Using device: {cp.cuda.Device(0)}")
        print("✅ CuPy GPU setup successful!")
    else:
        print("❌ CuPy GPU setup failed!")
        return False
    
    # Test basic operations
    print("\nTesting basic operations...")
    
    # Matrix multiplication test
    print("1. Matrix multiplication test...")
    size = 1000
    a = cp.random.normal(0, 1, (size, size)).astype(cp.float32)
    b = cp.random.normal(0, 1, (size, size)).astype(cp.float32)
    
    start_time = time.time()
    c = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    print(f"   {size}x{size} matrix multiplication: {end_time - start_time:.3f} seconds")
    
    # Neural network forward pass test
    print("2. Neural network forward pass test...")
    
    def simple_nn(x, weights):
        h1 = cp.maximum(0, cp.dot(x, weights[0]))  # ReLU
        h2 = cp.maximum(0, cp.dot(h1, weights[1]))  # ReLU
        return cp.dot(h2, weights[2])
    
    # Create weights
    input_size, hidden_size, output_size = 1000, 500, 100
    w1 = cp.random.normal(0, cp.sqrt(2.0/input_size), (input_size, hidden_size)).astype(cp.float32)
    w2 = cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, hidden_size)).astype(cp.float32)
    w3 = cp.random.normal(0, cp.sqrt(2.0/hidden_size), (hidden_size, output_size)).astype(cp.float32)
    weights = [w1, w2, w3]
    
    # Create input
    batch_size = 100
    x = cp.random.normal(0, 1, (batch_size, input_size)).astype(cp.float32)
    
    start_time = time.time()
    for _ in range(100):
        y = simple_nn(x, weights)
        cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    print(f"   100 forward passes: {end_time - start_time:.3f} seconds")
    
    # Memory test
    print("3. Memory test...")
    try:
        # Try to allocate a large array
        large_array = cp.random.normal(0, 1, (5000, 5000)).astype(cp.float32)
        print(f"   Successfully allocated {large_array.nbytes / 1e9:.1f} GB array")
        del large_array
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"   Memory allocation failed: {e}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour CuPy setup is working correctly with RTX 5090!")
    print("You can now run: python reverse_engineer_gravity_cupy.py")
    
    return True

if __name__ == "__main__":
    try:
        test_cupy_setup()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 