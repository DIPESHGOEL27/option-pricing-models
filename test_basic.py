#!/usr/bin/env python3
"""
Simple test to verify basic option pricing functionality
This will help us check if the deployment works correctly
"""

import sys
import os

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

def test_basic_functionality():
    """Test basic Black-Scholes pricing"""
    try:
        # Test basic imports
        import numpy as np
        import scipy.stats as si
        print(f"✅ NumPy {np.__version__} imported successfully")
        print(f"✅ SciPy imported successfully")
        
        # Test basic Black-Scholes calculation
        S = 100  # Stock price
        K = 100  # Strike price
        T = 1    # Time to expiration (1 year)
        r = 0.05 # Risk-free rate
        sigma = 0.2  # Volatility
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        put_price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
        
        print(f"✅ Call option price: ${call_price:.4f}")
        print(f"✅ Put option price: ${put_price:.4f}")
        
        # Test Flask app import
        from app import app
        print("✅ Flask app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing basic option pricing functionality...")
    success = test_basic_functionality()
    if success:
        print("✅ All basic tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
