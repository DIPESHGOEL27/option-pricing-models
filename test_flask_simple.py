#!/usr/bin/env python3
"""
Simple Flask app test to verify core functionality
"""

import sys
import os
sys.path.append('api')

try:
    print("Testing Flask app import...")
    from flask import Flask
    print("Flask imported successfully")
    
    print("Testing option pricing module...")
    from option_pricing import AdvancedOptionPricer
    pricer = AdvancedOptionPricer()
    result = pricer.black_scholes(100, 100, 0.25, 0.05, 0.2, 'call')
    print(f"Option pricing works: {result:.4f}")
    
    print("Testing core modules individually...")
    
    # Test basic imports
    try:
        from advanced_models import MonteCarloEngine
        print("✓ Advanced models imported")
    except Exception as e:
        print(f"✗ Advanced models error: {e}")
    
    try:
        from advanced_risk import AdvancedRiskManager
        print("✓ Advanced risk imported")
    except Exception as e:
        print(f"✗ Advanced risk error: {e}")
    
    try:
        from market_data_advanced import AdvancedMarketDataProvider
        print("✓ Market data advanced imported")
    except Exception as e:
        print(f"✗ Market data advanced error: {e}")
        
    try:
        from ml_pricing import NeuralNetworkPricer
        print("✓ ML pricing imported")
    except Exception as e:
        print(f"✗ ML pricing error: {e}")
        
    try:
        from model_validation import ModelValidator
        print("✓ Model validation imported")
    except Exception as e:
        print(f"✗ Model validation error: {e}")
        
    try:
        from portfolio_optimization import AdvancedPortfolioOptimizer
        print("✓ Portfolio optimization imported")
    except Exception as e:
        print(f"✗ Portfolio optimization error: {e}")
    
    print("\nAll core modules tested!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
