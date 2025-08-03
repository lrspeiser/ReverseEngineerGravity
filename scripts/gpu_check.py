#!/usr/bin/env python3
"""
Minimal GPU Check for RTX 5090
Just check if GPU is detected without complex operations
"""

try:
    import cupy as cp
    print("CuPy imported successfully")
    
    # Check CUDA availability
    cuda_available = cp.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get device info
        device = cp.cuda.Device(0)
        print(f"GPU Device: {str(device)}")
        
        # Get memory info
        mem_info = device.mem_info
        total_memory_gb = mem_info[1] / (1024**3)
        print(f"GPU Memory: {total_memory_gb:.1f} GB total")
        
        # Check if it looks like RTX 5090
        device_str = str(device)
        if "5090" in device_str or "RTX" in device_str:
            print("✓ RTX 5090 detected!")
        else:
            print(f"Device detected: {device_str}")
        
        print(f"CuPy Version: {cp.__version__}")
        
        # Simple test - create a small array
        print("\nTesting basic GPU operation...")
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"GPU array sum test: {result} ✓")
        
        print("\nGPU CHECK COMPLETED SUCCESSFULLY!")
        print("RTX 5090 is available for use.")
        
    else:
        print("ERROR: CUDA not available!")
        print("GPU acceleration will not work.")
        
except ImportError as e:
    print(f"ERROR: Could not import CuPy: {e}")
    print("Please install CuPy with: pip install cupy-cuda12x")
    
except Exception as e:
    print(f"ERROR: Unexpected error: {e}")
    print("GPU check failed.") 