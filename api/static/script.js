/**
 * Advanced Option Pricing Platform - JavaScript
 * Industry-grade frontend for sophisticated option pricing and risk management
 */

class OptionPricingPlatform {
  constructor() {
    this.portfolio = [];
    this.marketData = {};
    this.currentSymbol = "";
    this.init();
  }
  init() {
    this.setupEventListeners();
    this.setupAdvancedEventListeners();
    this.setupQuickActions();
    this.loadMarketDashboard();
    this.initializeDefaults();
    this.startRealTimeUpdates();

    // Initialize enhanced features
    this.initializeEnhancements();

    // Show welcome notification
    setTimeout(() => {
      this.showNotification(
        "Advanced Option Pricing Platform loaded successfully!",
        "success"
      );
    }, 1000);
  }

  initializeEnhancements() {
    // Add tooltips to all elements with title attribute
    const tooltipTriggerList = [].slice.call(
      document.querySelectorAll("[title]")
    );
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add click animations to buttons
    document.querySelectorAll(".btn").forEach((btn) => {
      btn.addEventListener("click", function (e) {
        let ripple = document.createElement("span");
        ripple.classList.add("ripple");
        this.appendChild(ripple);

        let x = e.clientX - e.target.offsetLeft;
        let y = e.clientY - e.target.offsetTop;

        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;

        setTimeout(() => {
          ripple.remove();
        }, 600);
      });
    });

    // Enhanced form validation
    this.setupFormValidation();
  }

  setupFormValidation() {
    // Add real-time validation to number inputs
    document.querySelectorAll('input[type="number"]').forEach((input) => {
      input.addEventListener("input", function () {
        if (this.value && parseFloat(this.value) <= 0) {
          this.classList.add("is-invalid");
        } else {
          this.classList.remove("is-invalid");
        }
      });
    });
  }

  setupEventListeners() {
    // Model type changes
    $("#modelType").on("change", this.onModelTypeChange.bind(this));

    // Calculation buttons with model state management
    $("#blackScholesBtn").on("click", () => {
      this.setActiveModelButton("blackScholesBtn");
      this.calculateBasic("black_scholes");
    });
    $("#binomialBtn").on("click", () => {
      this.setActiveModelButton("binomialBtn");
      this.calculateBasic("binomial");
    });
    $("#calculateAdvanced").on("click", this.calculateAdvanced.bind(this));

    // Market data
    $("#getDataBtn").on("click", this.getMarketData.bind(this));
    $("#symbolInput").on("keypress", (e) => {
      if (e.which === 13) this.getMarketData();
    });

    // Portfolio management
    $("#addPositionBtn").on("click", () =>
      $("#addPositionModal").modal("show")
    );
    $("#savePositionBtn").on("click", this.addPosition.bind(this));

    // Risk management
    $("#marketCrashBtn").on("click", () => this.runStressTest("market_crash"));
    $("#volSpikeBtn").on("click", () => this.runStressTest("vol_spike"));
    $("#rateShockBtn").on("click", () => this.runStressTest("rate_shock"));
    $("#validateModelsBtn").on("click", this.validateModels.bind(this));

    // Analysis buttons
    $("#payoffDiagramBtn").on("click", this.showPayoffDiagram.bind(this));
    $("#volatilitySmileBtn").on("click", this.showVolatilitySmile.bind(this));
    $("#greeksSensitivityBtn").on(
      "click",
      this.showGreeksSensitivity.bind(this)
    );
    $("#convergenceBtn").on("click", this.showConvergenceAnalysis.bind(this));
  }

  initializeDefaults() {
    // Set default values based on current market conditions
    const defaultParams = {
      S: 100,
      K: 100,
      T: 0.25,
      r: 0.05,
      sigma: 0.2,
    };

    Object.keys(defaultParams).forEach((key) => {
      $(`#${key}`).val(defaultParams[key]);
    });
  }

  async loadMarketDashboard() {
    try {
      const response = await fetch("/api/market_sentiment");

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.vix && !data.vix.error) {
        $("#vixLevel").text(data.vix.vix_level.toFixed(2));
        $("#vixSentiment").text(data.vix.sentiment);
        $("#fearGreedScore").text(data.vix.fear_greed_score.toFixed(0));
      } else {
        // Fallback values
        $("#vixLevel").text("--");
        $("#vixSentiment").text("Loading...");
        $("#fearGreedScore").text("--");
      }

      if (data.put_call_ratio && !data.put_call_ratio.error) {
        $("#putCallRatio").text(data.put_call_ratio.put_call_ratio.toFixed(2));
        $("#putCallSentiment").text(data.put_call_ratio.sentiment);
      } else {
        $("#putCallRatio").text("--");
        $("#putCallSentiment").text("Loading...");
      }

      if (data.treasury_rates && data.treasury_rates["10Y"]) {
        const rate10Y = data.treasury_rates["10Y"];
        if (!isNaN(rate10Y)) {
          $("#treasury10Y").text((rate10Y * 100).toFixed(2) + "%");
          $("#r").val(rate10Y.toFixed(4)); // Update risk-free rate
        }
      } else {
        $("#treasury10Y").text("--");
      }
    } catch (error) {
      console.error("Error loading market dashboard:", error);

      // Set default values on error
      $("#vixLevel").text("--");
      $("#vixSentiment").text("Error loading");
      $("#putCallRatio").text("--");
      $("#putCallSentiment").text("Error loading");
      $("#treasury10Y").text("--");
      $("#fearGreedScore").text("--");

      this.showNotification(
        "Warning: Unable to load real-time market data",
        "warning"
      );
    }
  }

  onModelTypeChange() {
    const modelType = $("#modelType").val();

    // Hide all parameter sections
    $("#hestonParams, #jumpParams").hide();

    // Show relevant parameters
    if (modelType === "heston") {
      $("#hestonParams").show();
    } else if (modelType === "jump_diffusion") {
      $("#jumpParams").show();
    }
  }

  async calculateBasic(model) {
    try {
      const params = this.getBasicParameters();
      let url, data;

      if (model === "black_scholes") {
        url = "/api/calculate_black_scholes";
        data = params;
        $("#binomialParams").hide();
      } else {
        url = "/api/calculate_binomial";
        data = { ...params, steps: parseInt($("#steps").val()) || 100 };
        $("#binomialParams").show();
      }

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();
      this.displayBasicResults(result, model);
    } catch (error) {
      console.error("Calculation error:", error);
      this.showAlert(
        "Error in calculation. Please check your inputs.",
        "danger"
      );
    }
  }

  async calculateAdvanced() {
    try {
      const params = this.getBasicParameters();
      const modelType = $("#modelType").val();
      const simulations = parseInt($("#simulations").val());

      const data = {
        ...params,
        model: modelType,
        simulations: simulations,
      };

      // Add model-specific parameters
      if (modelType === "heston") {
        data.kappa = parseFloat($("#kappa").val());
        data.theta = parseFloat($("#theta").val());
        data.sigma_v = parseFloat($("#sigma_v").val());
        data.rho = parseFloat($("#rho").val());
        data.v0 = data.theta; // Initial variance = long-term variance
      } else if (modelType === "jump_diffusion") {
        data.lambda = parseFloat($("#lambda").val());
        data.mu_j = parseFloat($("#mu_j").val());
        data.sigma_j = parseFloat($("#sigma_j").val());
      }

      const response = await fetch("/api/monte_carlo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();
      this.displayAdvancedResults(result);
    } catch (error) {
      console.error("Advanced calculation error:", error);
      this.showAlert("Error in advanced calculation.", "danger");
    }
  }

  async getMarketData() {
    const symbol = $("#symbolInput").val().toUpperCase();
    if (!symbol) return;

    try {
      const response = await fetch(`/api/market_data/${symbol}`);
      const data = await response.json();

      if (data.error) {
        this.showAlert(
          `Error fetching data for ${symbol}: ${data.error}`,
          "warning"
        );
        return;
      }

      this.currentSymbol = symbol;
      this.marketData[symbol] = data;
      this.displayMarketData(data);
      this.updateParametersFromMarketData(data);
    } catch (error) {
      console.error("Market data error:", error);
      this.showAlert("Error fetching market data.", "danger");
    }
  }

  addPosition() {
    try {
      const position = {
        symbol: $("#posSymbol").val().toUpperCase(),
        option_type: $("#posOptionType").val(),
        strike: parseFloat($("#posStrike").val()),
        quantity: parseInt($("#posQuantity").val()),
        premium_paid: parseFloat($("#posPremium").val()),
        expiry: $("#posExpiry").val(),
        underlying_price: parseFloat($("#posUnderlying").val()),
        volatility: parseFloat($("#posVolatility").val()),
        risk_free_rate: parseFloat($("#r").val()),
        id: Date.now(), // Simple ID for tracking
      };

      this.portfolio.push(position);
      this.updatePortfolioDisplay();
      $("#addPositionModal").modal("hide");
      $("#addPositionForm")[0].reset();

      this.showAlert("Position added successfully!", "success");
    } catch (error) {
      console.error("Error adding position:", error);
      this.showAlert(
        "Error adding position. Please check your inputs.",
        "danger"
      );
    }
  }

  removePosition(id) {
    this.portfolio = this.portfolio.filter((pos) => pos.id !== id);
    this.updatePortfolioDisplay();
    this.showAlert("Position removed.", "info");
  }

  async updatePortfolioDisplay() {
    if (this.portfolio.length === 0) {
      $("#portfolioTable tbody").html(`
                <tr>
                    <td colspan="7" class="text-center text-muted">
                        No positions in portfolio
                    </td>
                </tr>
            `);
      this.resetPortfolioSummary();
      return;
    }

    // Calculate portfolio metrics
    try {
      const response = await fetch("/api/risk_metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ positions: this.portfolio }),
      });

      const data = await response.json();

      // Update portfolio table
      let tableHTML = "";
      this.portfolio.forEach((pos) => {
        const posValue = pos.quantity * this.calculateCurrentPrice(pos);
        const pnl = posValue - pos.quantity * pos.premium_paid;

        tableHTML += `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td>${pos.option_type.toUpperCase()}</td>
                        <td>$${pos.strike}</td>
                        <td>${pos.quantity}</td>
                        <td>$${posValue.toFixed(2)}</td>
                        <td class="${
                          pnl >= 0 ? "text-success" : "text-danger"
                        }">
                            $${pnl.toFixed(2)}
                        </td>
                        <td>
                            <button class="btn btn-sm btn-danger" onclick="platform.removePosition(${
                              pos.id
                            })">
                                <i class="fas fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `;
      });

      $("#portfolioTable tbody").html(tableHTML);

      // Update portfolio summary
      if (data.portfolio_summary) {
        this.updatePortfolioSummary(data.portfolio_summary);
      }
    } catch (error) {
      console.error("Error updating portfolio:", error);
    }
  }

  calculateCurrentPrice(position) {
    // Simplified current price calculation
    // In a real application, this would use real-time option pricing
    return position.premium_paid * (0.9 + Math.random() * 0.2); // Mock price change
  }

  async runStressTest(scenario) {
    try {
      const params = this.getBasicParameters();
      const response = await fetch("/api/stress_test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      const result = await response.json();
      this.displayStressTestResults(result);
    } catch (error) {
      console.error("Stress test error:", error);
      this.showAlert("Error running stress test.", "danger");
    }
  }

  async validateModels() {
    try {
      const params = this.getBasicParameters();
      const response = await fetch("/api/model_validation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });

      const result = await response.json();
      this.displayValidationResults(result);
    } catch (error) {
      console.error("Model validation error:", error);
      this.showAlert("Error validating models.", "danger");
    }
  }

  async showPayoffDiagram() {
    if (this.portfolio.length === 0) {
      this.showAlert(
        "Please add positions to portfolio to view payoff diagram.",
        "warning"
      );
      return;
    }

    try {
      const response = await fetch("/api/plot_payoff", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ positions: this.portfolio }),
      });

      const result = await response.json();
      if (result.plot) {
        const plotData = JSON.parse(result.plot);
        Plotly.newPlot("analysisPlot", plotData.data, plotData.layout);
      }
    } catch (error) {
      console.error("Payoff diagram error:", error);
      this.showAlert("Error generating payoff diagram.", "danger");
    }
  }

  async showVolatilitySmile() {
    if (!this.currentSymbol) {
      this.showAlert("Please select a symbol first.", "warning");
      return;
    }

    try {
      const response = await fetch("/api/volatility_smile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: this.currentSymbol }),
      });

      const result = await response.json();
      if (result.plot) {
        const plotData = JSON.parse(result.plot);
        Plotly.newPlot("analysisPlot", plotData.data, plotData.layout);
      }
    } catch (error) {
      console.error("Volatility smile error:", error);
      this.showAlert("Error generating volatility smile.", "danger");
    }
  }

  showGreeksSensitivity() {
    // Create interactive Greeks sensitivity analysis
    const params = this.getBasicParameters();
    const spotRange = [];
    const deltaValues = [];
    const gammaValues = [];
    const vegaValues = [];

    // Generate sensitivity data
    for (let S = params.S * 0.7; S <= params.S * 1.3; S += params.S * 0.01) {
      spotRange.push(S);

      // Simplified Greeks calculations for demo
      const d1 =
        (Math.log(S / params.K) +
          (params.r + 0.5 * params.sigma ** 2) * params.T) /
        (params.sigma * Math.sqrt(params.T));

      deltaValues.push(this.normalCDF(d1));
      gammaValues.push(
        this.normalPDF(d1) / (S * params.sigma * Math.sqrt(params.T))
      );
      vegaValues.push(S * this.normalPDF(d1) * Math.sqrt(params.T));
    }

    const traces = [
      {
        x: spotRange,
        y: deltaValues,
        type: "scatter",
        mode: "lines",
        name: "Delta",
        line: { color: "blue" },
      },
      {
        x: spotRange,
        y: gammaValues,
        type: "scatter",
        mode: "lines",
        name: "Gamma",
        yaxis: "y2",
        line: { color: "red" },
      },
    ];

    const layout = {
      title: "Greeks Sensitivity Analysis",
      xaxis: { title: "Underlying Price" },
      yaxis: { title: "Delta", side: "left" },
      yaxis2: { title: "Gamma", side: "right", overlaying: "y" },
      template: "plotly_dark",
    };

    Plotly.newPlot("analysisPlot", traces, layout);
  }

  showConvergenceAnalysis() {
    // Monte Carlo convergence analysis
    const simulationSizes = [1000, 5000, 10000, 25000, 50000, 100000];
    const prices = [];
    const errors = [];

    // Simulate convergence (in real app, this would call the API)
    const truePrice = 10.45; // Mock true price
    simulationSizes.forEach((size) => {
      const error = (1 / Math.sqrt(size)) * 2; // Theoretical error reduction
      prices.push(truePrice + (Math.random() - 0.5) * error);
      errors.push(error);
    });

    const trace1 = {
      x: simulationSizes,
      y: prices,
      type: "scatter",
      mode: "lines+markers",
      name: "MC Price",
      line: { color: "green" },
    };

    const trace2 = {
      x: simulationSizes,
      y: errors,
      type: "scatter",
      mode: "lines+markers",
      name: "Standard Error",
      yaxis: "y2",
      line: { color: "orange" },
    };

    const layout = {
      title: "Monte Carlo Convergence Analysis",
      xaxis: { title: "Number of Simulations", type: "log" },
      yaxis: { title: "Option Price" },
      yaxis2: { title: "Standard Error", side: "right", overlaying: "y" },
      template: "plotly_dark",
    };

    Plotly.newPlot("analysisPlot", [trace1, trace2], layout);
  }

  // =================== ADVANCED FEATURES ===================

  async calculateMLPrice() {
    const data = {
      S: parseFloat($("#mlS").val()),
      K: parseFloat($("#mlK").val()),
      T: parseFloat($("#mlT").val()),
      r: parseFloat($("#mlR").val()),
      sigma: parseFloat($("#mlSigma").val()),
      optionType: $("#mlOptionType").val(),
    };

    try {
      const response = await fetch("/api/ml/ensemble_price", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      $("#mlResults").show();
      $("#mlPriceComparison").html(`
                <div class="row">
                    <div class="col-6">
                        <strong>ML Price:</strong> $${result.ml_price.toFixed(
                          4
                        )}<br>
                        <strong>BS Price:</strong> $${result.black_scholes_price.toFixed(
                          4
                        )}
                    </div>
                    <div class="col-6">
                        <strong>Difference:</strong> $${result.price_difference.toFixed(
                          4
                        )}<br>
                        <strong>Relative:</strong> ${result.relative_difference.toFixed(
                          2
                        )}%
                    </div>
                </div>
                <div class="mt-2">
                    <strong>Greeks:</strong> Δ=${result.greeks.delta.toFixed(
                      4
                    )}, 
                    Γ=${result.greeks.gamma.toFixed(4)}, 
                    Θ=${result.greeks.theta.toFixed(4)}
                </div>
            `);
    } catch (error) {
      this.showNotification(`ML Pricing Error: ${error.message}`, "danger");
    }
  }

  async optimizePortfolio() {
    const assets = $("#optAssets")
      .val()
      .split(",")
      .map((s) => s.trim());
    const data = {
      symbols: assets,
      method: $("#optMethod").val(),
      target_return: parseFloat($("#optTargetReturn").val()) / 100,
    };

    try {
      const response = await fetch("/api/portfolio/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      $("#optimizationResults").show();

      let weightsHtml = "<strong>Optimal Weights:</strong><br>";
      result.symbols.forEach((symbol, index) => {
        weightsHtml += `${symbol}: ${(
          result.optimal_weights[index] * 100
        ).toFixed(1)}%<br>`;
      });

      $("#optimalWeights").html(weightsHtml);
      $("#portfolioMetrics").html(`
                <strong>Expected Return:</strong> ${(
                  result.expected_return * 100
                ).toFixed(2)}%<br>
                <strong>Volatility:</strong> ${(
                  result.volatility * 100
                ).toFixed(2)}%<br>
                <strong>Sharpe Ratio:</strong> ${result.sharpe_ratio.toFixed(3)}
            `);
    } catch (error) {
      this.showNotification(
        `Portfolio Optimization Error: ${error.message}`,
        "danger"
      );
    }
  }

  async analyzeRisk() {
    // Mock portfolio positions for demonstration
    const positions = [
      { symbol: "AAPL", value: 100000 },
      { symbol: "GOOGL", value: 75000 },
      { symbol: "MSFT", value: 50000 },
    ];

    const data = {
      positions: positions,
      confidence_level: parseFloat($("#riskConfidence").val()) / 100,
      time_horizon: parseInt($("#riskHorizon").val()),
    };

    try {
      const response = await fetch("/api/risk/portfolio_risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      $("#riskResults").show();
      $("#riskMetricsDisplay").html(`
                <div class="row">
                    <div class="col-6">
                        <strong>Portfolio Value:</strong> $${result.portfolio_value.toLocaleString()}<br>
                        <strong>VaR (${(data.confidence_level * 100).toFixed(
                          0
                        )}%):</strong> $${Math.abs(
        result.var.historical
      ).toLocaleString()}<br>
                        <strong>Expected Shortfall:</strong> $${Math.abs(
                          result.expected_shortfall
                        ).toLocaleString()}
                    </div>
                    <div class="col-6">
                        <strong>Volatility:</strong> ${(
                          result.risk_metrics.volatility * 100
                        ).toFixed(1)}%<br>
                        <strong>Max Drawdown:</strong> ${(
                          result.risk_metrics.max_drawdown * 100
                        ).toFixed(1)}%<br>
                        <strong>Sharpe Ratio:</strong> ${result.risk_metrics.sharpe_ratio.toFixed(
                          3
                        )}
                    </div>
                </div>
                <div class="mt-2">
                    <strong>Stress Test Results:</strong><br>
                    Market Crash: ${(
                      result.stress_test_results.market_crash.loss * 100
                    ).toFixed(1)}% loss<br>
                    Vol Spike: ${(
                      result.stress_test_results.volatility_spike.loss * 100
                    ).toFixed(1)}% loss
                </div>
            `);
    } catch (error) {
      this.showNotification(`Risk Analysis Error: ${error.message}`, "danger");
    }
  }

  async calculateHedging() {
    const data = {
      portfolio_delta: parseFloat($("#portfolioDelta").val()),
      target_delta: 0,
      hedge_ratio: 1.0,
    };

    try {
      const response = await fetch("/api/risk/dynamic_hedging", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      $("#riskResults").show();
      $("#riskMetricsDisplay").html(`
                <div class="alert alert-info">
                    <strong>Hedging Strategy:</strong><br>
                    Current Delta: ${result.current_delta.toFixed(4)}<br>
                    Target Delta: ${result.target_delta.toFixed(4)}<br>
                    Delta Exposure: ${result.delta_exposure.toFixed(4)}<br>
                    Hedge Quantity: ${result.hedge_quantity.toFixed(
                      0
                    )} shares<br>
                    Recommendation: <strong>${result.recommendation.toUpperCase()}</strong><br>
                    Hedge Cost: $${result.hedge_cost.toFixed(2)}
                </div>
            `);
    } catch (error) {
      this.showNotification(
        `Hedging Calculation Error: ${error.message}`,
        "danger"
      );
    }
  }

  async getMarketSentiment() {
    const symbol = $("#sentimentSymbol").val() || "SPY";

    try {
      const response = await fetch(`/api/market/sentiment?symbol=${symbol}`);
      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      $("#sentimentResults").show();
      $("#fearGreedIndex").text(
        result.sentiment_indicators.fear_greed_index.toFixed(0)
      );
      $("#sentimentVix").text(result.sentiment_indicators.vix_level.toFixed(1));
      $("#putCallRatio").text(
        result.sentiment_indicators.put_call_ratio.toFixed(2)
      );
      $("#overallSentiment").text(result.overall_sentiment.toUpperCase());

      // Color coding based on sentiment
      const sentiment = result.overall_sentiment;
      $("#overallSentiment").removeClass(
        "text-success text-warning text-danger"
      );
      if (sentiment === "greedy") {
        $("#overallSentiment").addClass("text-success");
      } else if (sentiment === "fearful") {
        $("#overallSentiment").addClass("text-danger");
      } else {
        $("#overallSentiment").addClass("text-warning");
      }
    } catch (error) {
      this.showNotification(
        `Market Sentiment Error: ${error.message}`,
        "danger"
      );
    }
  }

  setupAdvancedEventListeners() {
    // ML Pricing
    $("#mlPriceBtn").on("click", () => this.calculateMLPrice());

    // Portfolio Optimization
    $("#optimizeBtn").on("click", () => this.optimizePortfolio());

    // Risk Management
    $("#riskAnalysisBtn").on("click", () => this.analyzeRisk());
    $("#hedgingBtn").on("click", () => this.calculateHedging());

    // Market Sentiment
    $("#sentimentBtn").on("click", () => this.getMarketSentiment());
  }

  // Utility methods
  getBasicParameters() {
    return {
      S: parseFloat($("#S").val()),
      K: parseFloat($("#K").val()),
      T: parseFloat($("#T").val()),
      r: parseFloat($("#r").val()),
      sigma: parseFloat($("#sigma").val()),
      optionType: $("#optionType").val(),
    };
  }

  displayBasicResults(result, model) {
    let html = `
            <div class="alert alert-success">
                <h6><i class="fas fa-check-circle me-2"></i>${model
                  .replace("_", "-")
                  .toUpperCase()} Results</h6>
            </div>
            <table class="table table-dark table-striped">
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
        `;

    Object.entries(result).forEach(([key, value]) => {
      if (typeof value === "number") {
        const formattedValue =
          key.includes("price") || key.includes("theta") || key.includes("rho")
            ? `$${value.toFixed(4)}`
            : value.toFixed(6);
        html += `<tr><td>${this.formatKey(
          key
        )}</td><td>${formattedValue}</td></tr>`;
      }
    });

    html += "</tbody></table>";
    $("#basicResults").html(html);
  }

  displayAdvancedResults(result) {
    let html = `
            <div class="alert alert-info">
                <h6><i class="fas fa-rocket me-2"></i>Monte Carlo Results</h6>
                <small>Model: ${
                  result.model_type?.toUpperCase() || "Unknown"
                } | 
                       Simulations: ${
                         result.simulations?.toLocaleString() || "Unknown"
                       }</small>
            </div>
            <table class="table table-dark table-striped table-sm">
        `;

    const metrics = {
      "Option Price": result.option_price,
      "Std Error": result.std_error,
      Delta: result.delta,
      Gamma: result.gamma,
      Vega: result.vega,
      Theta: result.theta,
    };

    Object.entries(metrics).forEach(([key, value]) => {
      if (value !== undefined) {
        const formattedValue =
          typeof value === "number"
            ? key.includes("Price") || key.includes("Theta")
              ? `$${value.toFixed(4)}`
              : value.toFixed(6)
            : value;
        html += `<tr><td>${key}</td><td>${formattedValue}</td></tr>`;
      }
    });

    if (result.confidence_interval) {
      html += `<tr><td>95% CI</td><td>[$${result.confidence_interval[0].toFixed(
        4
      )}, $${result.confidence_interval[1].toFixed(4)}]</td></tr>`;
    }

    html += "</table>";
    $("#advancedResults").html(html);
  }

  displayExoticResults(result) {
    let html = `
            <div class="alert alert-warning">
                <h6><i class="fas fa-star me-2"></i>Exotic Option Results</h6>
                <small>Type: ${
                  result.exotic_type?.toUpperCase() || "Unknown"
                }</small>
            </div>
            <table class="table table-dark table-striped table-sm">
                <tr><td>Option Price</td><td>$${
                  result.option_price?.toFixed(4) || "N/A"
                }</td></tr>
                <tr><td>Standard Error</td><td>${
                  result.std_error?.toFixed(6) || "N/A"
                }</td></tr>
        `;

    // Add exotic-specific metrics
    if (result.barrier_hit_prob) {
      html += `<tr><td>Barrier Hit Probability</td><td>${(
        result.barrier_hit_prob * 100
      ).toFixed(2)}%</td></tr>`;
    }
    if (result.hit_probability) {
      html += `<tr><td>Hit Probability</td><td>${(
        result.hit_probability * 100
      ).toFixed(2)}%</td></tr>`;
    }

    html += "</table>";
    $("#exoticResults").html(html);
  }

  displayMarketData(data) {
    const tableHTML = `
            <tr><td>Price</td><td>$${data.price?.toFixed(2) || "N/A"}</td></tr>
            <tr><td>Change</td><td class="${
              data.day_change >= 0 ? "text-success" : "text-danger"
            }">
                ${data.day_change?.toFixed(2) || "N/A"} (${
      data.day_change_pct?.toFixed(2) || "N/A"
    }%)
            </td></tr>
            <tr><td>Volume</td><td>${
              data.volume?.toLocaleString() || "N/A"
            }</td></tr>
            <tr><td>IV</td><td>${
              (data.implied_volatility * 100)?.toFixed(1) || "N/A"
            }%</td></tr>
        `;

    $("#symbolDataTable").html(tableHTML);
    $("#symbolData").removeClass("d-none");
  }

  updateParametersFromMarketData(data) {
    if (data.price) $("#S").val(data.price.toFixed(2));
    if (data.implied_volatility)
      $("#sigma").val(data.implied_volatility.toFixed(3));
  }

  displayStressTestResults(result) {
    let html = `
            <div class="alert alert-danger">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>Stress Test Results</h6>
            </div>
            <table class="table table-dark table-striped table-sm">
                <tr><td>Base Price</td><td>$${result.base_price?.toFixed(
                  4
                )}</td></tr>
        `;

    Object.entries(result).forEach(([scenario, data]) => {
      if (scenario !== "base_price" && typeof data === "object") {
        const pnlClass = data.pnl_impact >= 0 ? "text-success" : "text-danger";
        html += `
                    <tr>
                        <td>${scenario.replace("_", " ").toUpperCase()}</td>
                        <td class="${pnlClass}">
                            $${data.pnl_impact?.toFixed(
                              4
                            )} (${data.pnl_percent?.toFixed(2)}%)
                        </td>
                    </tr>
                `;
      }
    });

    html += "</table>";
    $("#stressResults").html(html);
  }

  displayValidationResults(result) {
    const validation = result.validation;
    let html = `
            <div class="alert alert-${
              validation.is_within_confidence ? "success" : "warning"
            }">
                <h6><i class="fas fa-check-circle me-2"></i>Model Validation</h6>
                <small>Validation: ${
                  validation.is_within_confidence ? "PASSED" : "REVIEW NEEDED"
                }</small>
            </div>
            <table class="table table-dark table-striped table-sm">
                <tr><td>Black-Scholes Price</td><td>$${validation.black_scholes_price?.toFixed(
                  4
                )}</td></tr>
                <tr><td>Monte Carlo Price</td><td>$${validation.monte_carlo_price?.toFixed(
                  4
                )}</td></tr>
                <tr><td>Absolute Error</td><td>$${validation.absolute_error?.toFixed(
                  6
                )}</td></tr>
                <tr><td>Relative Error</td><td>${(
                  validation.relative_error * 100
                )?.toFixed(4)}%</td></tr>
            </table>
        `;

    $("#stressResults").html(html);
  }

  updatePortfolioSummary(summary) {
    $("#totalValue").text(`$${summary.portfolio_value?.toFixed(2) || "0.00"}`);
    $("#totalPnL").text(`$${summary.total_pnl?.toFixed(2) || "0.00"}`);

    if (summary.portfolio_greeks) {
      $("#portfolioDelta").text(
        summary.portfolio_greeks.delta?.toFixed(4) || "0.00"
      );
      $("#portfolioGamma").text(
        summary.portfolio_greeks.gamma?.toFixed(6) || "0.00"
      );
      $("#portfolioVega").text(
        summary.portfolio_greeks.vega?.toFixed(4) || "0.00"
      );
      $("#portfolioTheta").text(
        `$${summary.portfolio_greeks.theta?.toFixed(2) || "0.00"}`
      );
    }
  }

  resetPortfolioSummary() {
    $("#totalValue, #totalPnL").text("$0.00");
    $("#portfolioDelta, #portfolioGamma, #portfolioVega").text("0.00");
    $("#portfolioTheta").text("$0.00");
  }

  formatKey(key) {
    return key
      .replace(/_/g, " ")
      .replace(/\b\w/g, (l) => l.toUpperCase())
      .replace("Option Price", "Price")
      .replace("Std Error", "Std. Error");
  }

  showAlert(message, type = "info") {
    const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show position-fixed" 
                 style="top: 80px; right: 20px; z-index: 9999; min-width: 300px;">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

    $("body").append(alertHTML);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      $(".alert").fadeOut();
    }, 5000);
  }

  // Enhanced UI Management
  showLoading() {
    document.getElementById("loadingOverlay").classList.add("active");
  }

  hideLoading() {
    document.getElementById("loadingOverlay").classList.remove("active");
  }

  showNotification(message, type = "info", duration = 5000) {
    const container = document.getElementById("notificationContainer");
    const notification = document.createElement("div");
    notification.className = `alert alert-${type} alert-dismissible fade show notification-item`;
    notification.style.marginBottom = "10px";
    notification.innerHTML = `
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    container.appendChild(notification);

    // Auto remove after duration
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, duration);
  }

  // Enhanced Real-time Market Data
  async updateMarketDashboard() {
    try {
      this.showLoading();

      // Update VIX data
      const vixResponse = await fetch("/api/market_data/^VIX");
      if (vixResponse.ok) {
        const vixData = await vixResponse.json();
        if (vixData.price && !isNaN(vixData.price)) {
          document.getElementById("vixLevel").textContent =
            vixData.price.toFixed(2);

          const vixSentiment =
            vixData.price < 20
              ? "Low Fear"
              : vixData.price < 30
              ? "Moderate Fear"
              : "High Fear";
          document.getElementById("vixSentiment").textContent = vixSentiment;
        }
      }

      // Update Treasury data
      const treasuryResponse = await fetch("/api/market_data/^TNX");
      if (treasuryResponse.ok) {
        const treasuryData = await treasuryResponse.json();
        if (treasuryData.price && !isNaN(treasuryData.price)) {
          document.getElementById("treasury10Y").textContent =
            treasuryData.price.toFixed(2) + "%";
        }
      }

      this.hideLoading();
    } catch (error) {
      console.error("Market dashboard update failed:", error);
      this.showNotification("Failed to update market data", "warning");
      this.hideLoading();
    }
  }

  // Enhanced Option Pricing with Animations
  async calculateBasicWithAnimation(model) {
    const formData = this.getFormData();
    if (!this.validateFormData(formData)) return;

    this.showLoading();

    try {
      const response = await fetch(`/api/calculate_${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const result = await response.json();
      this.displayResultsWithAnimation(result, model);
      this.showNotification(
        `${model} calculation completed successfully`,
        "success"
      );
    } catch (error) {
      this.showNotification(`Calculation failed: ${error.message}`, "danger");
    } finally {
      this.hideLoading();
    }
  }

  displayResultsWithAnimation(result, model) {
    const resultsDiv = document.getElementById("basicResults");

    // Create enhanced results display
    const resultsHTML = `
      <div class="card glass-card">
        <div class="card-header">
          <h6 class="mb-0">
            <i class="fas fa-chart-line me-2"></i>${model
              .replace("_", "-")
              .toUpperCase()} Results
          </h6>
        </div>
        <div class="card-body">
          <div class="row">
            <div class="col-md-6">
              <div class="metric-card">
                <div class="metric-label">Option Price</div>
                <div class="metric-value">${
                  result.option_price?.toFixed(4) || "N/A"
                }</div>
              </div>
            </div>
            <div class="col-md-6">
              <div class="metric-card">
                <div class="metric-label">Delta</div>
                <div class="metric-value performance-${
                  result.delta > 0 ? "positive" : "negative"
                }">
                  ${result.delta?.toFixed(4) || "N/A"}
                </div>
              </div>
            </div>
          </div>
          <div class="row mt-3">
            <div class="col-md-3">
              <div class="performance-indicator">
                <div class="performance-label">Gamma</div>
                <div class="performance-value">${
                  result.gamma?.toFixed(4) || "N/A"
                }</div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="performance-indicator">
                <div class="performance-label">Vega</div>
                <div class="performance-value">${
                  result.vega?.toFixed(4) || "N/A"
                }</div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="performance-indicator">
                <div class="performance-label">Theta</div>
                <div class="performance-value performance-${
                  result.theta < 0 ? "negative" : "positive"
                }">
                  ${result.theta?.toFixed(4) || "N/A"}
                </div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="performance-indicator">
                <div class="performance-label">Rho</div>
                <div class="performance-value">${
                  result.rho?.toFixed(4) || "N/A"
                }</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    resultsDiv.innerHTML = resultsHTML;

    // Add entrance animation
    const cards = resultsDiv.querySelectorAll(
      ".metric-card, .performance-indicator"
    );
    cards.forEach((card, index) => {
      card.style.opacity = "0";
      card.style.transform = "translateY(20px)";

      setTimeout(() => {
        card.style.transition = "all 0.5s ease";
        card.style.opacity = "1";
        card.style.transform = "translateY(0)";
      }, index * 100);
    });
  }

  // Enhanced Portfolio Management
  async addPositionWithValidation() {
    const position = {
      symbol: document.getElementById("posSymbol").value.toUpperCase(),
      optionType: document.getElementById("posOptionType").value,
      strike: parseFloat(document.getElementById("posStrike").value),
      quantity: parseInt(document.getElementById("posQuantity").value),
      premium: parseFloat(document.getElementById("posPremium").value),
      expiry: document.getElementById("posExpiry").value,
      underlying: parseFloat(document.getElementById("posUnderlying").value),
      volatility: parseFloat(document.getElementById("posVolatility").value),
    };

    // Validate position data
    if (!this.validatePosition(position)) {
      this.showNotification(
        "Please fill in all required fields correctly",
        "danger"
      );
      return;
    }

    this.showLoading();

    try {
      // Calculate current option value
      const pricingResponse = await fetch("/api/calculate_black_scholes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          S: position.underlying,
          K: position.strike,
          T: this.calculateTimeToExpiry(position.expiry),
          r: 0.05, // Default risk-free rate
          sigma: position.volatility,
          optionType: position.optionType,
        }),
      });

      if (pricingResponse.ok) {
        const pricingResult = await pricingResponse.json();
        position.currentValue =
          pricingResult.option_price * position.quantity * 100;
        position.pnl =
          position.currentValue - position.premium * position.quantity * 100;
        position.delta = pricingResult.delta * position.quantity * 100;
        position.gamma = pricingResult.gamma * position.quantity * 100;
        position.vega = pricingResult.vega * position.quantity;
        position.theta = pricingResult.theta * position.quantity;
      }

      this.portfolio.push(position);
      this.updatePortfolioDisplay();
      this.showNotification(
        `Position ${position.symbol} added successfully`,
        "success"
      );

      // Close modal and reset form
      $("#addPositionModal").modal("hide");
      document.getElementById("addPositionForm").reset();
    } catch (error) {
      this.showNotification(
        `Failed to add position: ${error.message}`,
        "danger"
      );
    } finally {
      this.hideLoading();
    }
  }

  validatePosition(position) {
    return (
      position.symbol &&
      position.strike > 0 &&
      position.quantity !== 0 &&
      position.premium > 0 &&
      position.underlying > 0 &&
      position.volatility > 0 &&
      position.expiry
    );
  }

  calculateTimeToExpiry(expiryDate) {
    const today = new Date();
    const expiry = new Date(expiryDate);
    const timeDiff = expiry.getTime() - today.getTime();
    return Math.max(0, timeDiff / (1000 * 3600 * 24 * 365)); // Years
  }

  // Enhanced Real-time Updates
  startRealTimeUpdates() {
    // Update market dashboard every 30 seconds
    this.marketUpdateInterval = setInterval(() => {
      this.updateMarketDashboard();
    }, 30000);

    // Update portfolio every 60 seconds
    this.portfolioUpdateInterval = setInterval(() => {
      if (this.portfolio.length > 0) {
        this.updatePortfolioValues();
      }
    }, 60000);
  }

  stopRealTimeUpdates() {
    if (this.marketUpdateInterval) {
      clearInterval(this.marketUpdateInterval);
    }
    if (this.portfolioUpdateInterval) {
      clearInterval(this.portfolioUpdateInterval);
    }
  }

  // Enhanced Quick Actions
  setupQuickActions() {
    const fab = document.getElementById("quickActionFab");
    const menu = document.getElementById("quickActionMenu");

    fab.addEventListener("click", () => {
      const isVisible = menu.style.display !== "none";
      menu.style.display = isVisible ? "none" : "block";
      fab.innerHTML = isVisible
        ? '<i class="fas fa-plus"></i>'
        : '<i class="fas fa-times"></i>';
    });

    // Quick price calculation
    document.getElementById("quickPrice").addEventListener("click", () => {
      // Auto-fill with market data and calculate
      document.getElementById("S").value = "100";
      document.getElementById("K").value = "100";
      document.getElementById("T").value = "0.25";
      document.getElementById("r").value = "0.05";
      document.getElementById("sigma").value = "0.2";
      this.calculateBasicWithAnimation("black_scholes");
      menu.style.display = "none";
      fab.innerHTML = '<i class="fas fa-plus"></i>';
    });

    // Quick portfolio addition
    document.getElementById("quickPortfolio").addEventListener("click", () => {
      $("#addPositionModal").modal("show");
      menu.style.display = "none";
      fab.innerHTML = '<i class="fas fa-plus"></i>';
    });

    // Quick risk check
    document.getElementById("quickRisk").addEventListener("click", () => {
      document.getElementById("risk-tab").click();
      menu.style.display = "none";
      fab.innerHTML = '<i class="fas fa-plus"></i>';
    });
  }

  // Mathematical utility functions
  normalCDF(x) {
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  normalPDF(x) {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  }

  erf(x) {
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y =
      1.0 -
      ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  setActiveModelButton(activeButtonId) {
    // Remove active class from all model buttons and add inactive
    $(".model-btn").removeClass("active").addClass("inactive");

    // Remove any existing pulse animation
    $(".model-btn").removeClass("pulse-animation");

    // Add active class to the clicked button and remove inactive
    $(`#${activeButtonId}`).removeClass("inactive").addClass("active");

    // Add pulse animation to the newly selected button
    $(`#${activeButtonId}`).addClass("pulse-animation");

    // Remove pulse animation after it completes
    setTimeout(() => {
      $(`#${activeButtonId}`).removeClass("pulse-animation");
    }, 800);

    // Show notification about model selection
    const modelName =
      activeButtonId === "blackScholesBtn" ? "Black-Scholes" : "Binomial";
    this.showNotification(`${modelName} model selected`, "info", 2000);
  }
}

// Initialize the platform when the page loads
let platform;
$(document).ready(function () {
  platform = new OptionPricingPlatform();

  // Set up periodic market data refresh
  setInterval(() => {
    platform.loadMarketDashboard();
  }, 300000); // Refresh every 5 minutes
});

// Global functions for HTML onclick events
window.platform = platform;
