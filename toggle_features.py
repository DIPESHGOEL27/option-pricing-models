#!/usr/bin/env python3
"""
Feature toggle script to enable/disable heavy dependencies
Usage: python toggle_features.py [enable|disable|minimal]
"""

import sys
import os

def create_requirements(mode="full"):
    """Create requirements.txt based on deployment mode"""
    
    core_deps = [
        "Flask>=2.3.0,<4.0.0",
        "numpy>=1.24.0,<2.0.0", 
        "scipy>=1.9.0,<2.0.0",
        "pandas>=1.5.0,<3.0.0",
        "requests>=2.31.0,<3.0.0",
        "gunicorn>=20.1.0,<22.0.0"
    ]
    
    light_deps = [
        "plotly>=5.17.0,<6.0.0",
        "yfinance>=0.2.0,<1.0.0",
        "matplotlib>=3.6.0,<4.0.0"
    ]
    
    heavy_deps = [
        "scikit-learn>=1.1.0,<2.0.0",
        "seaborn>=0.12.0,<1.0.0", 
        "statsmodels>=0.13.0,<1.0.0",
        "joblib>=1.2.0,<2.0.0",
        "aiohttp>=3.8.0,<4.0.0",
        "cvxpy>=1.2.0,<2.0.0"
    ]
    
    requirements = core_deps.copy()
    
    if mode == "minimal":
        # Only core dependencies
        pass
    elif mode == "light":
        requirements.extend(light_deps)
    elif mode == "full":
        requirements.extend(light_deps)
        requirements.extend(heavy_deps)
    
    return requirements

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if mode not in ["minimal", "light", "full"]:
        print("Usage: python toggle_features.py [minimal|light|full]")
        sys.exit(1)
    
    requirements = create_requirements(mode)
    
    with open("requirements.txt", "w") as f:
        f.write(f"# Generated requirements for {mode} deployment\n")
        f.write("# Use toggle_features.py to change deployment mode\n\n")
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"✅ Updated requirements.txt for {mode} deployment")
    print(f"📦 {len(requirements)} packages included")
    
    if mode == "minimal":
        print("⚠️  Only basic features available (Black-Scholes, Greeks)")
    elif mode == "light": 
        print("📈 Basic + plotting features available")
    else:
        print("🚀 All features available (may exceed some deployment limits)")

if __name__ == "__main__":
    main()
