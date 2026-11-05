# Monte Carlo Price Simulation

Simulates future stock and cryptocurrency prices with Monte Carlo methods, using historical data from Yahoo Finance.

**Try it in the browser:** [Monte Carlo Simulator](https://boonwui.github.io/monte-carlo/)

## Models

| Script | Model |
|---|---|
| `monte carlo simulation gbm.py` | **Geometric Brownian motion:** constant drift and volatility estimated from history. |
| `monte carlo simulation heston.py` | **Heston stochastic volatility:** volatility itself moves randomly and reverts to a long-run level. This gives fatter tails than GBM. |
| `monte carlo simulation stock.py` | **Normal daily returns:** each day's return is drawn from a normal distribution with the historical mean and standard deviation. |

For each ticker, the scripts:

1. download price history with `yfinance`,
2. simulate thousands of price paths over 252 trading days (one year),
3. plot the distribution of final prices and the expected percentage return for each ticker.

## Setup and run

```bash
pip install yfinance numpy matplotlib
python "monte carlo simulation heston.py"
```

Edit the ticker list at the bottom of each script.

## Disclaimer

For education and research only. This is not financial advice. Simulations are based on historical data and models, which may not reflect future performance.

Part of [BoonWui's portfolio](https://boonwui.github.io).
