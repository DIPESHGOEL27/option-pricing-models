# Project Structure

```
e:\Option_Pricing_Models/
├── .git/                           # Git repository
├── .gitignore                      # Git ignore rules
├── .python-version                 # Python version for Railway
├── __init__.py                     # Makes root a Python package
├── main.py                         # Entry point for Railway deployment
├── setup.py                        # Python package setup
├── requirements.txt                # Python dependencies
├── requirements-vercel.txt         # Lightweight deps for Vercel
├── runtime.txt                     # Python version for Vercel
├── 
├── # Deployment Configurations
├── Dockerfile                      # For Google Cloud Run / Docker
├── Procfile                        # For Railway/Heroku
├── railway.toml                    # Railway-specific config
├── railway.json                    # Railway build config
├── nixpacks.toml                   # Nixpacks build config
├── vercel.json                     # Vercel deployment config
├── 
├── # Documentation
├── README.md                       # Main project documentation
├── DEPLOYMENT_GUIDE.md             # General deployment guide
├── RAILWAY_DEPLOY.md               # Railway deployment instructions
├── CLOUDRUN_DEPLOY.md              # Google Cloud Run instructions
├── 
├── # Utilities
├── config.py                       # Configuration settings
├── toggle_features.py              # Feature toggling for deployment
├── deploy_helper.py                # Deployment utilities
├── 
├── # Testing
├── test_basic.py                   # Basic functionality test
├── test_compatibility.py           # Compatibility test suite
├── test_suite.py                   # Comprehensive test suite
├── simple_test.py                  # Simple validation test
├── test_flask_simple.py           # Flask-specific test
├── 
├── # Main Application (api/)
└── api/
    ├── __init__.py                 # API package marker
    ├── app.py                      # Main Flask application
    ├── 
    ├── # Core Pricing Models
    ├── option_pricing.py           # Core option pricing algorithms
    ├── advanced_models.py          # Advanced pricing models (Monte Carlo, Heston)
    ├── 
    ├── # Advanced Features
    ├── ml_pricing.py               # Machine learning pricing models
    ├── advanced_risk.py            # Advanced risk management
    ├── portfolio_optimization.py   # Portfolio optimization algorithms
    ├── model_validation.py         # Model validation and backtesting
    ├── 
    ├── # Market Data
    ├── market_data.py              # Basic market data functionality
    ├── market_data_advanced.py     # Advanced market data and sentiment
    ├── portfolio_management.py     # Portfolio management utilities
    ├── 
    ├── # Frontend
    ├── templates/
    │   └── index.html              # Main web interface
    ├── static/
    │   ├── styles.css              # CSS styling
    │   └── script.js               # JavaScript functionality
    ├── 
    └── __pycache__/                # Python bytecode cache
```

## Key Files for Deployment:

### Railway Deployment:
- `main.py` - Entry point
- `railway.toml` - Railway configuration
- `nixpacks.toml` - Build configuration
- `requirements.txt` - Dependencies

### Vercel Deployment:
- `api/app.py` - Flask app
- `vercel.json` - Vercel configuration
- `requirements-vercel.txt` - Lightweight dependencies

### Google Cloud Run:
- `Dockerfile` - Container configuration
- `requirements.txt` - Dependencies

### Core Application:
- `api/app.py` - Main Flask application with all endpoints
- `api/option_pricing.py` - Core Black-Scholes and advanced pricing
- `api/advanced_models.py` - Monte Carlo, Heston, exotic options
- `api/ml_pricing.py` - Neural networks and ML models
- `api/templates/index.html` - Web interface
