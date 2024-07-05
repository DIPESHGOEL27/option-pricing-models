# **Options Pricing Models**

### **Introduction**

Objective:
To understand and implement option pricing models using Python.
To analyze the impact of different parameters on option pricing.

Importance:
Option pricing models are fundamental in financial derivatives, providing crucial insights for traders, investors, and risk managers.
Accurate pricing models help in making informed decisions, hedging risks, and maximizing returns.
<br><br>
Basics of Options:
* Call Option: The right to buy an asset at a specified price within a specified time.
* Put Option: The right to sell an asset at a specified price within a specified time.
<br><br>

### **The Black-Scholes Model**

In finance, the Black-Scholes-Merton Model is one of the most widely used methods for pricing options. It calculates the theoritical value of an option based on five key variables:

**Key Parameters**:

*   Underlying Asset Price (S)
*   Strike Price (K)
*   Time to Expiration or Maturity (T)
*   Risk - Free Interest Rate (r)
*   Volatility (σ)


### Black-Scholes Assumptions

The Black-Scholes model makes certain assumptions:

* No dividends are paid out during the life of the option.
* Markets are random (i.e., market movements cannot be predicted).
* There are no transaction costs in buying the option.
* The risk-free rate and volatility of the underlying asset are known and constant.
* The returns of the underlying asset are normally distributed.
* The option is European and can only be exercised at expiration.

### Option's Greeks
* An option's Greeks describe its various risk parameters.
* Delta is a measure of the change in an option's price or premium resulting from a change in the underlying asset, while theta measures its price decay as time passes.
* Gamma measures the delta's rate of change over time, as well as the rate of change in the underlying asset, and helps forecast price moves in the underlying asset.
* Vega measures the risk of changes in implied volatility or the forward-looking expected volatility of the underlying asset price.
* Theta measures time decay in the value of an option or its premium.

### Profitability
* An at-the-money option means that the option's strike price and the underlying asset's price are equal
* An in-the-money option means that a profit exists because the option's strike price is more favorable than the underlying asset's price
* An out-of-the-money (OTM) option means that no profit exists when comparing the option's strike price to the price of the underlying asset

### Binomial Options Pricing Model

The Binomial Options Pricing Model, introduced by Cox, Ross, and Rubinstein in 1979, provides a discrete-time framework for valuing options. It models the underlying asset price movements through a binomial tree, where each node represents potential price changes (up or down) over time.

#### Key Features
* **Discrete Time Steps:** Divides the option's life into multiple periods.
* **Binomial Tree:** Represents possible future asset prices at each step.
* **Flexibility:** Can handle American options and varying conditions.

#### Comparison with Black-Scholes Model
* **Time Framework:** Binomial is discrete, while Black-Scholes is continuous.
* **Flexibility:** Binomial can price American options; Black-Scholes primarily handles European options.
* **Complexity:** Binomial is computationally intensive for many steps; Black-Scholes provides a closed-form solution.

The Binomial Model's flexibility makes it a powerful tool for various option types and market conditions, complementing the more straightforward Black-Scholes Model.


