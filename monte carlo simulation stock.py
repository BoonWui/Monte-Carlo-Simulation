# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:31:06 2025

@author: BWLAU
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate stock price paths using Monte Carlo
def monte_carlo_simulation(ticker, num_simulations, num_days, initial_price):
    # Fetch historical data for the ticker using yfinance
    data = yf.download(ticker, period="10y", interval="1d")
    
    # Calculate the daily returns and volatility
    #data['daily_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['daily_return'] = data['Adj Close'].pct_change()
    daily_return_mean = data['daily_return'].mean()
    daily_return_std = data['daily_return'].std()
    
    # Simulate stock price paths
    simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        price_path = [initial_price]
        for j in range(1, num_days):
            random_return = np.random.normal(daily_return_mean, daily_return_std)
            price_path.append(price_path[-1] * (1 + random_return))
        simulations[i] = price_path
    
    # Calculate the expected value (mean of all simulated paths)
    expected_value = np.mean(simulations[:, -1])
    
    return simulations, expected_value

# Function to calculate and display the frequency distribution
def display_frequency_distribution(simulations, num_bins=10):
    # Extract the final prices of all simulations
    final_prices = simulations[:, -1]
    
    # Create bins for the final prices
    min_price = np.min(final_prices)
    max_price = np.max(final_prices)
    bin_edges = np.linspace(min_price, max_price, num_bins + 1)
    
    # Calculate frequencies for each bin
    frequencies, _ = np.histogram(final_prices, bins=bin_edges)
    
    # Display the frequency distribution
    print("\nPrice Range | Frequency")
    for i in range(len(frequencies)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} | {frequencies[i]}")
    
    # Plot the frequency distribution
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=bin_edges, edgecolor='black', alpha=0.7)
    plt.title("Frequency Distribution of Simulated Stock Prices")
    plt.xlabel("Price ($)")
    plt.ylabel("Frequency")
    plt.show()

def get_latest_price(ticker):
    # Fetch historical data for the ticker using yfinance
    data = yf.download(ticker, period="1d", interval="1d")
    latest_price = data['Adj Close'][-1]  # Get the latest adjusted closing price
    return latest_price

# List of tickers to analyze
tickers = ["AMZN", "MSFT", "GOOGL", "AAPL", "ELV","TSLA","BTC-USD","AVAX-USD","NVDA","ETH-usd","^IXIC","^GSPC","^DJI","SOL-USD","ADA-USD"]  # You can add more tickers here
#tickers=["AVAX-usd"]
# Initialize empty lists to store the results
tickers_list = []
current_prices = []
expected_prices = []
percentage_returns = []

# Run the Monte Carlo Simulation for each ticker
for ticker in tickers:
    # Get the latest price for the ticker
    latest_price = get_latest_price(ticker)
    
    # Set parameters for the simulation
    num_simulations = 5000  # Number of simulations to run
    num_days = 252          # Number of days to simulate (e.g., 252 for a year of trading days)
    
    # Run the Monte Carlo Simulation
    simulations, expected_value = monte_carlo_simulation(ticker, num_simulations, num_days, latest_price)
    
    # Calculate the percentage return
    percentage_return = (expected_value - latest_price) / latest_price * 100
    
    # Store the results
    tickers_list.append(ticker)
    current_prices.append(latest_price)
    expected_prices.append(expected_value)
    percentage_returns.append(percentage_return)
    
    # Display the results
    print(f"\nCurrent price for {ticker} is: ${latest_price:.2f}")
    print(f"Expected price for {ticker} after {num_days} days: ${expected_value:.2f}")
    print(f"Percentage return for {ticker} after {num_days} days: {percentage_return:.2f}%")
    
    # Display the frequency distribution
    display_frequency_distribution(simulations)

# Plot the percentage returns vs tickers
plt.figure(figsize=(10, 6))
plt.bar(tickers_list, percentage_returns, color='skyblue', edgecolor='black')
plt.title('Percentage Return of Simulated Stock Prices After 252 Days')
plt.xlabel('Ticker')
plt.ylabel('Percentage Return (%)')
plt.xticks(rotation=45)
plt.show()
