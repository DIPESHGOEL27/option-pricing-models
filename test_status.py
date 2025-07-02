#!/usr/bin/env python3
"""
Test the API status endpoint to identify issues
"""

import sys
sys.path.append('api')

try:
    from app import app
    print("✓ App imported successfully")
    
    # Test the status route
    with app.test_client() as client:
        print("Testing /api/status endpoint...")
        response = client.get('/api/status')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.get_json()
            print("✓ Status endpoint working")
            print(f"Features available: {len(data.get('features', {}))}")
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
            print(f"Response: {response.get_data(as_text=True)}")
            
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
