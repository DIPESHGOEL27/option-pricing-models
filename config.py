# Advanced Option Pricing Platform - Configuration

# Flask Configuration
DEBUG = False
SECRET_KEY = 'your-secret-key-change-in-production'

# Market Data Settings
MARKET_DATA_CACHE_DURATION = 300  # 5 minutes
YAHOO_FINANCE_TIMEOUT = 10
MAX_SYMBOLS_PER_REQUEST = 10

# Monte Carlo Settings
DEFAULT_SIMULATIONS = 100000
MAX_SIMULATIONS = 1000000
DEFAULT_STEPS = 252
MAX_STEPS = 1000

# Risk Management
VaR_CONFIDENCE_LEVELS = [0.95, 0.99]
STRESS_TEST_SCENARIOS = {
    'market_crash': {'S_shock': -0.3, 'vol_shock': 2.0},
    'vol_spike': {'S_shock': 0.0, 'vol_shock': 1.5},
    'rate_shock': {'r_shock': 0.02},
    'combined_stress': {'S_shock': -0.2, 'vol_shock': 1.8, 'r_shock': 0.01}
}

# Portfolio Limits
MAX_PORTFOLIO_POSITIONS = 100
POSITION_SIZE_LIMIT = 1000000  # $1M per position

# API Rate Limiting
REQUESTS_PER_MINUTE = 100
BURST_LIMIT = 200

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Database (if using)
# DATABASE_URL = 'postgresql://user:password@localhost/options_db'

# Cache Configuration
REDIS_URL = 'redis://localhost:6379/0'
CACHE_TYPE = 'simple'  # or 'redis' for production

# Security Headers
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}

# CORS Settings
CORS_ORIGINS = ['http://localhost:3000', 'https://yourdomain.com']

# Model Validation Settings
MODEL_VALIDATION_TOLERANCE = 0.01  # 1% tolerance
CONVERGENCE_ANALYSIS_POINTS = [1000, 5000, 10000, 25000, 50000, 100000]

# Default Option Parameters
DEFAULT_OPTION_PARAMS = {
    'S': 100.0,
    'K': 100.0,
    'T': 0.25,
    'r': 0.05,
    'sigma': 0.2
}
