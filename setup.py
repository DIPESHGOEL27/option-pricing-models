from setuptools import setup, find_packages

setup(
    name="option-pricing-models",
    version="1.0.0",
    description="Advanced Option Pricing Platform with ML and Risk Management",
    author="DIPESH GOEL",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "Flask>=2.3.0,<4.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.9.0,<2.0.0",
        "pandas>=1.5.0,<3.0.0",
        "requests>=2.31.0,<3.0.0",
        "gunicorn>=20.1.0,<22.0.0",
        "plotly>=5.17.0,<6.0.0",
        "yfinance>=0.2.0,<1.0.0",
        "matplotlib>=3.6.0,<4.0.0",
    ],
    entry_points={
        'console_scripts': [
            'option-pricing-server=main:main',
        ],
    },
)
