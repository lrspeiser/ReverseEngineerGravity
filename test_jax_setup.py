#!/usr/bin/env python3
"""
Test script to verify JAX setup with GPU support.
"""

import jax
import jax.numpy as jnp
import time

def test_jax_gpu():
    """Test JAX GPU functionality."""
    print("="*50)
    print("JAX GPU SETUP TEST")
    print("="*50)
    
    # Check JAX version
    print(f"JAX version: {jax.__version__}")
    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    if not devices:
        print("❌ No devices found!")
        return False
    
    # Check if GPU is available
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        print(f"✅ GPU found: {gpu_devices[0]}")
        device = gpu_devices[0]
    else:
        print("⚠️  No GPU found, using CPU")
        device = devices[0]
    
    # Test basic operations
    print("\nTesting basic operations...")
    
    # Create test arrays
    size = 10000
    a = jnp.random.normal(jax.random.PRNGKey(0), (size, size))
    b = jnp.random.normal(jax.random.PRNGKey(1), (size, size))
    
    # Time matrix multiplication
    start_time = time.time()
    c = jnp.dot(a, b)
    c.block_until_ready()  # Ensure computation is complete
    end_time = time.time()
    
    print(f"Matrix multiplication ({size}x{size}): {end_time - start_time:.3f} seconds")
    
    # Test neural network operations
    print("\nTesting neural network operations...")
    
    # Simple neural network forward pass
    def simple_nn(x, w1, w2):
        h = jax.nn.relu(jnp.dot(x, w1))
        return jnp.dot(h, w2)
    
    # Create weights
    input_size = 1000
    hidden_size = 500
    output_size = 10
    batch_size = 100
    
    w1 = jnp.random.normal(jax.random.PRNGKey(2), (input_size, hidden_size))
    w2 = jnp.random.normal(jax.random.PRNGKey(3), (hidden_size, output_size))
    x = jnp.random.normal(jax.random.PRNGKey(4), (batch_size, input_size))
    
    # Time forward pass
    start_time = time.time()
    for _ in range(100):
        y = simple_nn(x, w1, w2)
        y.block_until_ready()
    end_time = time.time()
    
    print(f"100 forward passes: {end_time - start_time:.3f} seconds")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    
    def loss_fn(w1, w2, x, target):
        pred = simple_nn(x, w1, w2)
        return jnp.mean((pred - target) ** 2)
    
    target = jnp.random.normal(jax.random.PRNGKey(5), (batch_size, output_size))
    
    # Compute gradients
    start_time = time.time()
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grads = grad_fn(w1, w2, x, target)
    grads[0].block_until_ready()
    grads[1].block_until_ready()
    end_time = time.time()
    
    print(f"Gradient computation: {end_time - start_time:.3f} seconds")
    
    print("\n✅ JAX GPU setup test completed successfully!")
    return True

if __name__ == "__main__":
    test_jax_gpu() 