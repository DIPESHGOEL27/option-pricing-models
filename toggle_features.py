#!/usr/bin/env python3
"""
Feature toggle script to generate requirements.txt for a given deployment mode.
Usage: python toggle_features.py [minimal|full]

There is no longer a "heavy" ML dependency tier -- ML pricing (scikit-learn,
xgboost, joblib) was removed as dead weight (it predicted Black-Scholes
prices from Black-Scholes-generated data, so it could at best rediscover a
closed-form formula this project already computes exactly). "full" is now
just core + market data/plotting deps.
"""

import sys

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

    full_deps = [
        "plotly>=5.17.0,<6.0.0",
        "yfinance>=0.2.0,<1.0.0",
        "jugaad-data>=0.35.0,<1.0.0"
    ]

    requirements = core_deps.copy()

    if mode == "full":
        requirements.extend(full_deps)

    return requirements

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode not in ["minimal", "full"]:
        print("Usage: python toggle_features.py [minimal|full]")
        sys.exit(1)

    requirements = create_requirements(mode)

    with open("requirements.txt", "w") as f:
        f.write(f"# Generated requirements for {mode} deployment\n")
        f.write("# Use toggle_features.py to change deployment mode\n\n")
        for req in requirements:
            f.write(f"{req}\n")

    print(f"Updated requirements.txt for {mode} deployment")
    print(f"{len(requirements)} packages included")

    if mode == "minimal":
        print("Only Black-Scholes, Binomial, and Monte Carlo pricing available (no market data, no charts)")
    else:
        print("All features available: pricing, market data, NSE option chains, charts")

if __name__ == "__main__":
    main()
