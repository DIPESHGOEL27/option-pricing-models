#!/usr/bin/env python3
"""
Compatibility Test Suite for Option Pricing Platform

This script tests the core functionality of all modules to ensure
they work correctly despite NumPy/SciPy compatibility warnings
on Python 3.13 Windows builds.
"""

import sys
import traceback
import warnings
from typing import Dict, List, Any

# Suppress NumPy warnings for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*MINGW-W64.*')

def test_result(test_name: str, success: bool, details: str = "") -> Dict[str, Any]:
    """Create a standardized test result"""
    return {
        'test': test_name,
        'success': success,
        'details': details
    }

def test_basic_imports() -> List[Dict[str, Any]]:
    """Test basic library imports"""
    results = []
    
    # Test NumPy
    try:
        import numpy as np
        results.append(test_result("NumPy Import", True, f"Version: {np.__version__}"))
    except Exception as e:
        results.append(test_result("NumPy Import", False, str(e)))
    
    # Test SciPy
    try:
        import scipy
        results.append(test_result("SciPy Import", True, f"Version: {scipy.__version__}"))
    except Exception as e:
        results.append(test_result("SciPy Import", False, str(e)))
    
    # Test Flask
    try:
        import flask
        results.append(test_result("Flask Import", True, f"Version: {flask.__version__}"))
    except Exception as e:
        results.append(test_result("Flask Import", False, str(e)))
        
    return results

def test_core_option_pricing() -> List[Dict[str, Any]]:
    """Test core option pricing functionality"""
    results = []
    
    try:
        # Import core modules
        sys.path.append('./api')
        import option_pricing
        
        # Test basic Black-Scholes calculation
        price = option_pricing.black_scholes(100, 105, 0.25, 0.05, 0.2, 'call')
        if isinstance(price, (int, float)) and price > 0:
            results.append(test_result("Black-Scholes Calculation", True, f"Call price: ${price:.2f}"))
        else:
            results.append(test_result("Black-Scholes Calculation", False, f"Invalid price: {price}"))
        
        # Test Greeks calculation
        greeks = option_pricing.calculate_greeks(100, 105, 0.25, 0.05, 0.2, 'call')
        if isinstance(greeks, dict) and 'delta' in greeks:
            results.append(test_result("Greeks Calculation", True, f"Delta: {greeks['delta']:.4f}"))
        else:
            results.append(test_result("Greeks Calculation", False, "Invalid Greeks output"))
            
        # Test Advanced Option Pricer
        pricer = option_pricing.AdvancedOptionPricer()
        advanced_price = pricer.black_scholes(100, 105, 0.25, 0.05, 0.2, 'call')
        if isinstance(advanced_price, (int, float)) and advanced_price > 0:
            results.append(test_result("Advanced Pricer", True, f"Price: ${advanced_price:.2f}"))
        else:
            results.append(test_result("Advanced Pricer", False, f"Invalid price: {advanced_price}"))
            
    except Exception as e:
        results.append(test_result("Option Pricing Module", False, f"Error: {str(e)}"))
        traceback.print_exc()
    
    return results

def test_advanced_modules() -> List[Dict[str, Any]]:
    """Test advanced modules"""
    results = []
    
    # Test ML Pricing
    try:
        import ml_pricing
        results.append(test_result("ML Pricing Import", True, "Module imported successfully"))
        
        # Test basic ML model creation
        ml_pricer = ml_pricing.MLOptionPricer()
        results.append(test_result("ML Pricer Instantiation", True, "MLOptionPricer created"))
        
    except Exception as e:
        results.append(test_result("ML Pricing Module", False, str(e)))
    
    # Test Advanced Risk
    try:
        import advanced_risk
        risk_manager = advanced_risk.AdvancedRiskManager()
        results.append(test_result("Risk Manager", True, "AdvancedRiskManager created"))
    except Exception as e:
        results.append(test_result("Advanced Risk Module", False, str(e)))
    
    # Test Market Data
    try:
        import market_data_advanced
        market_data = market_data_advanced.AdvancedMarketDataManager()
        results.append(test_result("Market Data Manager", True, "Manager created"))
    except Exception as e:
        results.append(test_result("Market Data Module", False, str(e)))
    
    # Test Model Validation
    try:
        import model_validation
        validator = model_validation.ModelValidator()
        results.append(test_result("Model Validator", True, "Validator created"))
    except Exception as e:
        results.append(test_result("Model Validation Module", False, str(e)))
    
    # Test Portfolio Optimization
    try:
        import portfolio_optimization
        optimizer = portfolio_optimization.PortfolioOptimizer()
        results.append(test_result("Portfolio Optimizer", True, "Optimizer created"))
    except Exception as e:
        results.append(test_result("Portfolio Optimization Module", False, str(e)))
    
    return results

def test_numerical_calculations() -> List[Dict[str, Any]]:
    """Test core numerical calculations"""
    results = []
    
    try:
        import numpy as np
        from scipy.stats import norm
        
        # Test basic NumPy operations
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        
        if abs(mean_val - 3.0) < 1e-10:
            results.append(test_result("NumPy Basic Operations", True, f"Mean: {mean_val}, Std: {std_val:.4f}"))
        else:
            results.append(test_result("NumPy Basic Operations", False, f"Mean calculation error: {mean_val}"))
        
        # Test SciPy normal distribution
        cdf_val = norm.cdf(0)
        if abs(cdf_val - 0.5) < 1e-10:
            results.append(test_result("SciPy Normal CDF", True, f"norm.cdf(0) = {cdf_val}"))
        else:
            results.append(test_result("SciPy Normal CDF", False, f"CDF error: {cdf_val}"))
            
    except Exception as e:
        results.append(test_result("Numerical Calculations", False, str(e)))
    
    return results

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("OPTION PRICING PLATFORM - COMPATIBILITY TEST SUITE")
    print("=" * 80)
    print()
    
    all_results = []
    
    # Run all test suites
    test_suites = [
        ("Basic Imports", test_basic_imports),
        ("Numerical Calculations", test_numerical_calculations),
        ("Core Option Pricing", test_core_option_pricing),
        ("Advanced Modules", test_advanced_modules)
    ]
    
    for suite_name, test_func in test_suites:
        print(f"Running {suite_name} Tests...")
        print("-" * 50)
        
        try:
            results = test_func()
            all_results.extend(results)
            
            for result in results:
                status = "✓ PASS" if result['success'] else "✗ FAIL"
                print(f"{status:8} | {result['test']:30} | {result['details']}")
                
        except Exception as e:
            print(f"✗ FAIL   | {suite_name:30} | Test suite failed: {str(e)}")
            all_results.append(test_result(f"{suite_name} Suite", False, str(e)))
        
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests:  {total_tests}")
    print(f"Passed:       {passed_tests}")
    print(f"Failed:       {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print()
        print("FAILED TESTS:")
        print("-" * 40)
        for result in all_results:
            if not result['success']:
                print(f"• {result['test']}: {result['details']}")
    
    print()
    print("=" * 80)
    
    return passed_tests, failed_tests

if __name__ == "__main__":
    passed, failed = run_comprehensive_test()
    
    # Return appropriate exit code
    sys.exit(0 if failed == 0 else 1)
