# Advanced Option Pricing Platform

## Overview

This is an industry-grade, professional option pricing and risk management platform that implements sophisticated quantitative finance models. The platform provides comprehensive tools for pricing various types of options, managing portfolios, conducting risk analysis, and performing advanced analytics.

## Features

### 🎯 Core Pricing Models
- **Black-Scholes Model**: Classical European option pricing with full Greeks
- **Binomial Tree Model**: Discrete-time pricing with American exercise capability
- **Monte Carlo Simulation**: Advanced stochastic modeling with multiple processes
- **Heston Stochastic Volatility**: Industry-standard stochastic volatility modeling
- **Jump-Diffusion Models**: Merton jump-diffusion for handling market jumps

### 🌟 Exotic Options
- **Asian Options**: Arithmetic and geometric average price options
- **Barrier Options**: Knock-in/knock-out options with various barrier types
- **Lookback Options**: Path-dependent options tracking extremes
- **Binary/Digital Options**: All-or-nothing payoff structures

### 📊 Portfolio Management
- **Multi-Asset Portfolios**: Support for complex option strategies
- **Real-time Greek Calculations**: Delta, Gamma, Vega, Theta, Rho
- **Risk Attribution**: Component-wise risk breakdown
- **Hedge Recommendations**: Automated delta-hedging suggestions

### 🛡️ Risk Management
- **Stress Testing**: Market crash, volatility spike, rate shock scenarios
- **Value at Risk (VaR)**: Statistical risk measures with confidence intervals
- **Model Validation**: Cross-validation between pricing models
- **Monte Carlo Convergence**: Statistical accuracy analysis

### 📈 Market Data Integration
- **Real-time Price Feeds**: Live stock and option data via Yahoo Finance
- **Volatility Surfaces**: Implied volatility analysis across strikes and expirations
- **Market Sentiment**: VIX, Put/Call ratios, Fear & Greed indicators
- **Risk-free Rates**: US Treasury yield curve integration

### 🔬 Advanced Analytics
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Payoff Diagrams**: Strategy visualization and P&L analysis
- **Volatility Smile**: Market implied volatility patterns
- **Greeks Sensitivity**: Dynamic sensitivity analysis
- **Correlation Analysis**: Multi-asset correlation and covariance

## Technical Architecture

### Backend (Python/Flask)
```
api/
├── app.py                    # Main Flask application
├── advanced_models.py        # Monte Carlo, Heston, Exotic options
├── portfolio_management.py   # Portfolio and risk management
├── market_data.py           # Real-time data integration
└── option_pricing.py        # Basic Black-Scholes implementation
```

### Frontend (Modern Web)
```
static/
├── script.js               # Advanced JavaScript platform
└── styles.css              # Modern dark theme UI

templates/
└── index.html              # Responsive single-page application
```

### Key Dependencies
- **NumPy/SciPy**: Scientific computing and statistics
- **Pandas**: Data manipulation and analysis
- **yfinance**: Real-time market data
- **Plotly**: Interactive visualizations
- **Flask**: Web framework
- **Bootstrap 5**: Modern responsive UI

## Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Option_Pricing_Models
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Development Server
```bash
cd api
python app.py
```

### 4. Access Platform
Navigate to `http://localhost:5000` in your web browser.

## Usage Guide

### Basic Option Pricing
1. Navigate to the **Basic Models** tab
2. Enter option parameters (S, K, T, r, σ)
3. Select Black-Scholes or Binomial model
4. Click calculate to see price and Greeks

### Advanced Monte Carlo
1. Go to **Advanced Models** tab
2. Choose model type (GBM, Heston, Jump-Diffusion)
3. Set simulation parameters
4. Configure model-specific parameters if needed
5. Run advanced calculation

### Exotic Options
1. Select **Exotic Options** tab
2. Choose exotic option type
3. Configure type-specific parameters
4. Calculate exotic option price

### Portfolio Management
1. Open **Portfolio** tab
2. Click "Add Position" to create positions
3. View portfolio summary and Greeks
4. Monitor real-time P&L

### Risk Analysis
1. Use **Risk Management** tab
2. Run stress tests with predefined scenarios
3. Validate models for accuracy
4. Review risk metrics and recommendations

### Market Data
1. Enter symbol in market data search
2. View real-time price and volatility
3. Parameters auto-populate from market data
4. Monitor market sentiment dashboard

## API Endpoints

### Core Pricing
- `POST /api/calculate_black_scholes` - Black-Scholes pricing
- `POST /api/calculate_binomial` - Binomial tree pricing
- `POST /api/monte_carlo` - Monte Carlo simulation
- `POST /api/exotic_options` - Exotic option pricing

### Market Data
- `GET /api/market_data/<symbol>` - Real-time stock data
- `GET /api/option_chain/<symbol>` - Option chain data
- `GET /api/volatility_surface/<symbol>` - Implied volatility surface
- `GET /api/market_sentiment` - Market sentiment indicators

### Risk & Portfolio
- `POST /api/risk_metrics` - Portfolio risk analysis
- `POST /api/stress_test` - Stress testing scenarios
- `POST /api/model_validation` - Model validation tests

### Visualization
- `POST /api/plot_payoff` - Payoff diagram generation
- `POST /api/volatility_smile` - Volatility smile charts

## Model Specifications

### Black-Scholes Model
```
Call Price = S₀N(d₁) - Ke^(-rT)N(d₂)
Put Price = Ke^(-rT)N(-d₂) - S₀N(-d₁)

Where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Heston Stochastic Volatility
```
dS = rSdt + √V S dW₁
dV = κ(θ - V)dt + σᵥ√V dW₂

Where:
κ = mean reversion speed
θ = long-term variance
σᵥ = volatility of volatility
ρ = correlation between price and volatility
```

### Monte Carlo Framework
- **Variance Reduction**: Antithetic variates, control variates
- **Convergence Analysis**: Statistical accuracy monitoring
- **Multiple Processes**: GBM, Heston, Jump-diffusion support
- **Parallel Processing**: Optimized for performance

## Risk Management Framework

### Greeks Calculation
- **Delta**: Price sensitivity to underlying
- **Gamma**: Delta sensitivity (convexity)
- **Vega**: Volatility sensitivity
- **Theta**: Time decay
- **Rho**: Interest rate sensitivity

### Risk Metrics
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Expected Shortfall**: Conditional VaR
- **Maximum Drawdown**: Historical peak-to-trough loss
- **Sharpe Ratio**: Risk-adjusted returns

### Stress Testing
- **Market Crash**: -30% equity shock
- **Volatility Spike**: +50% volatility increase
- **Rate Shock**: +200bp interest rate change
- **Custom Scenarios**: User-defined stress tests

## Performance Optimizations

### Computational Efficiency
- **Vectorized Operations**: NumPy-optimized calculations
- **Caching**: Market data and calculation caching
- **Batch Processing**: Multiple option calculations
- **Memory Management**: Efficient array operations

### User Experience
- **Real-time Updates**: Live market data integration
- **Responsive Design**: Mobile and desktop compatibility
- **Progressive Loading**: Asynchronous data fetching
- **Error Handling**: Comprehensive error management

## Deployment

### Production Deployment (Vercel)
The platform is configured for deployment on Vercel with:
- `vercel.json` configuration
- Python runtime support
- Automatic HTTPS
- Global CDN distribution

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python api/app.py

# Access at http://localhost:5000
```

### Environment Variables
```bash
# Optional: Set API keys for enhanced market data
export ALPHA_VANTAGE_API_KEY=your_key
export QUANDL_API_KEY=your_key
```

## Contributing

### Code Structure
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters
- Implement comprehensive error handling
- Add docstrings for all functions

### Testing
- Unit tests for all pricing models
- Integration tests for API endpoints
- Performance benchmarks
- Model validation tests

### Documentation
- Update README for new features
- Document API changes
- Include mathematical formulations
- Provide usage examples

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes. It should not be used for actual trading without proper validation and risk management. The authors are not responsible for any financial losses incurred through the use of this software.

## Support

For questions, issues, or contributions:
1. Open an issue on GitHub
2. Submit a pull request
3. Contact the development team

---

**Built with ❤️ for the quantitative finance community**
