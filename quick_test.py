#!/usr/bin/env python3
"""
Quick verification test for the Option Pricing Platform
Tests basic functionality to ensure the app is working before full verification
"""

import requests
import sys

def test_basic_functionality(base_url="http://127.0.0.1:8000"):
    """Test basic app functionality"""
    print("🔍 Quick Functionality Test")
    print("=" * 40)
    
    try:
        # Test 1: Main page
        print("1. Testing main page...")
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ Main page accessible")
        else:
            print(f"   ❌ Main page failed: {response.status_code}")
            return False
        
        # Test 2: API status
        print("2. Testing API status...")
        response = requests.get(f"{base_url}/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API status accessible")
            print(f"   📊 Features available: {len(data.get('features', {}))}")
        else:
            print(f"   ❌ API status failed: {response.status_code}")
            return False
        
        # Test 3: Black-Scholes pricing
        print("3. Testing Black-Scholes pricing...")
        pricing_data = {
            "S": 100,
            "K": 100,
            "T": 0.25,
            "r": 0.05,
            "sigma": 0.2,
            "optionType": "call"
        }
        
        response = requests.post(
            f"{base_url}/api/calculate_black_scholes",
            json=pricing_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            price = data.get('option_price', 0)
            print(f"   ✅ Black-Scholes pricing: ${price:.4f}")
        else:
            print(f"   ❌ Black-Scholes failed: {response.status_code}")
            return False
        
        # Test 4: Performance metrics
        print("4. Testing performance metrics...")
        response = requests.get(f"{base_url}/api/performance_metrics", timeout=30)
        if response.status_code == 200:
            data = response.json()
            capacity = data.get('options_per_day_capacity', 0)
            print(f"   ✅ Performance metrics: {capacity:,} options/day")
        else:
            print(f"   ❌ Performance metrics failed: {response.status_code}")
            return False
        
        print("\n🎉 All basic tests passed! App is ready for full verification.")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to {base_url}")
        print("   💡 Make sure the Flask app is running: python main.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if not success:
        print("\n❌ Basic tests failed.")
    
    sys.exit(0 if success else 1)
