#!/usr/bin/env python3
"""
Simple test to isolate the NumPy issue on Python 3.13 Windows
"""

print("Starting compatibility test...")

try:
    print("Testing basic Python functionality...")
    import sys
    print(f"Python version: {sys.version}")
    
    print("Testing NumPy import...")
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print("NumPy imported successfully!")
    
    print("Testing basic NumPy operations...")
    arr = np.array([1, 2, 3, 4, 5])
    mean_val = float(np.mean(arr))
    print(f"Array mean: {mean_val}")
    
    print("Testing SciPy import...")
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    
    from scipy.stats import norm
    cdf_val = float(norm.cdf(0))
    print(f"Normal CDF(0): {cdf_val}")
    
    print("All basic tests passed!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
