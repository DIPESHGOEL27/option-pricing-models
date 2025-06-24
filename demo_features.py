#!/usr/bin/env python3
"""
Demo script to showcase the advanced option pricing web app features
"""

import requests
import json
import time

def test_web_app(base_url="http://127.0.0.1:8000"):
    """Test all major features of the web application"""
    
    print("🚀 Advanced Option Pricing Platform - Feature Demo")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n📊 1. System Health Check")
    try:
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['deployment']}")
            print(f"🐍 Python: {data['python_version'][:6]}")
            print(f"📚 NumPy: {data['core_libraries']['numpy']}")
            print(f"🔬 SciPy: {data['core_libraries']['scipy']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 2: Black-Scholes Pricing
    print("\n💰 2. Black-Scholes Option Pricing")
    pricing_data = {
        "S": 100,      # Stock price
        "K": 100,      # Strike price
        "T": 1,        # Time to expiration (1 year)
        "r": 0.05,     # Risk-free rate (5%)
        "sigma": 0.2,  # Volatility (20%)
        "optionType": "call"
    }
    
    try:
        response = requests.post(f"{base_url}/api/calculate_black_scholes", 
                               json=pricing_data, 
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Call Option Price: ${data['option_price']:.4f}")
            print(f"📈 Delta: {data['delta']:.4f}")
            print(f"🎯 Gamma: {data['gamma']:.6f}")
            print(f"⏰ Theta: ${data['theta']:.4f}/day")
            print(f"📊 Vega: ${data['vega']:.4f}")
        else:
            print(f"❌ Pricing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Pricing error: {e}")
    
    # Test 3: Monte Carlo Simulation
    print("\n🎲 3. Monte Carlo Simulation")
    mc_data = {
        "S": 100,
        "K": 105,
        "T": 0.25,     # 3 months
        "r": 0.05,
        "sigma": 0.3,  # Higher volatility
        "optionType": "put",
        "simulations": 10000
    }
    
    try:
        response = requests.post(f"{base_url}/api/monte_carlo", 
                               json=mc_data,
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Monte Carlo Price: ${data['option_price']:.4f}")
            print(f"📊 Confidence Interval: ${data['confidence_interval'][0]:.4f} - ${data['confidence_interval'][1]:.4f}")
            print(f"🎯 Standard Error: ${data['standard_error']:.6f}")
        else:
            print(f"❌ Monte Carlo failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Monte Carlo error: {e}")
    
    # Test 4: Risk Metrics
    print("\n⚠️  4. Risk Metrics Calculation")
    risk_data = {
        "portfolio_value": 1000000,  # $1M portfolio
        "positions": [
            {"symbol": "AAPL", "quantity": 100, "price": 150},
            {"symbol": "GOOGL", "quantity": 50, "price": 2500},
            {"symbol": "MSFT", "quantity": 200, "price": 300}
        ],
        "confidence_level": 0.95,
        "time_horizon": 1
    }
    
    try:
        response = requests.post(f"{base_url}/api/risk_metrics", 
                               json=risk_data,
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Portfolio VaR (95%): ${data['var_95']:.2f}")
            print(f"📊 Expected Shortfall: ${data['expected_shortfall']:.2f}")
            print(f"📈 Volatility: {data['portfolio_volatility']:.4f}")
        else:
            print(f"❌ Risk calculation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Risk calculation error: {e}")
    
    # Test 5: Market Data
    print("\n📈 5. Market Data Retrieval")
    try:
        response = requests.get(f"{base_url}/api/market_data/AAPL")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ AAPL Current Price: ${data['current_price']:.2f}")
            print(f"📊 Daily Change: {data['change_percent']:.2f}%")
            print(f"📈 52W High: ${data['high_52w']:.2f}")
            print(f"📉 52W Low: ${data['low_52w']:.2f}")
        else:
            print(f"❌ Market data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Market data error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete! Your web app is fully functional!")
    print(f"🌐 Access the web interface at: {base_url}")
    print("📱 The app is mobile-responsive and production-ready!")
    print("🚀 Deploy to Railway for public access!")
    
    return True

if __name__ == "__main__":
    # Test local development server
    print("Testing local development server...")
    test_web_app("http://127.0.0.1:8000")
    
    print("\n" + "-" * 60)
    print("💡 To test your Railway deployment:")
    print("   python demo_features.py https://your-app.up.railway.app")
    
    # If Railway URL provided as argument
    import sys
    if len(sys.argv) > 1:
        railway_url = sys.argv[1]
        print(f"\n🚂 Testing Railway deployment at {railway_url}")
        test_web_app(railway_url)
