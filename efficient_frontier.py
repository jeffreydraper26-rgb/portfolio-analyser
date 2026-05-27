import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fredapi import Fred

# ---- SETUP ---- #
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ---- SETTINGS ---- #
tickers = ["SPY", "JPM", "META", "V"]
start = "2015-01-01"
end = "2025-01-01"
n_portfolios = 10000

# ---- DOWNLOAD MONTHLY PRICE DATA ---- #
prices = yf.download(tickers, start=start, end=end, interval="1mo")["Close"]

# ---- CALCULATE MONTHLY RETURNS ---- #
monthly_returns = prices.pct_change().dropna()

# ---- ANNUALISE RETURNS AND COVARIANCE ---- #
ann_returns = (1 + monthly_returns.mean()) ** 12 - 1
ann_cov = monthly_returns.cov() * 12

# ---- GET RISK FREE RATE (most recent 3 month T-bill) ---- #
tbill = fred.get_series("TB3MS", observation_start=start, observation_end=end)
risk_free_rate = tbill.iloc[-1] / 100

# ---- GENERATE RANDOM PORTFOLIOS ---- #
results = np.zeros((n_portfolios, 3))
weights_record = []

for i in range(n_portfolios):
    w = np.random.random(len(tickers))
    w = w / w.sum()
    weights_record.append(w)

    ret = np.dot(w, ann_returns)
    vol = np.sqrt(w @ ann_cov @ w)
    sharpe = (ret - risk_free_rate) / vol

    results[i] = [ret, vol, sharpe]

results_df = pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe"])
weights_df = pd.DataFrame(weights_record, columns=tickers)

# ---- FIND KEY PORTFOLIOS ---- #
max_sharpe_idx = results_df["Sharpe"].idxmax()
min_vol_idx = results_df["Volatility"].idxmin()

max_sharpe = results_df.loc[max_sharpe_idx]
min_vol = results_df.loc[min_vol_idx]

print("\nRisk Free Rate (most recent 3M T-Bill):")
print(f"  {risk_free_rate:.2%}")

print("\nMaximum Sharpe Ratio Portfolio:")
print(f"  Return:     {max_sharpe['Return']:.2%}")
print(f"  Volatility: {max_sharpe['Volatility']:.2%}")
print(f"  Sharpe:     {max_sharpe['Sharpe']:.4f}")
print("  Weights:")
for ticker, w in zip(tickers, weights_df.loc[max_sharpe_idx]):
    print(f"    {ticker}: {w:.2%}")

print("\nMinimum Variance Portfolio:")
print(f"  Return:     {min_vol['Return']:.2%}")
print(f"  Volatility: {min_vol['Volatility']:.2%}")
print(f"  Sharpe:     {min_vol['Sharpe']:.4f}")
print("  Weights:")
for ticker, w in zip(tickers, weights_df.loc[min_vol_idx]):
    print(f"    {ticker}: {w:.2%}")

# ---- YOUR EQUAL WEIGHT PORTFOLIO ---- #
your_weights = np.array([0.25, 0.25, 0.25, 0.25])
your_return = np.dot(your_weights, ann_returns)
your_vol = np.sqrt(your_weights @ ann_cov @ your_weights)
your_sharpe = (your_return - risk_free_rate) / your_vol

print("\nYour Equal Weight Portfolio:")
print(f"  Return:     {your_return:.2%}")
print(f"  Volatility: {your_vol:.2%}")
print(f"  Sharpe:     {your_sharpe:.4f}")

# ---- SPY PORTFOLIO ---- #
spy_return = ann_returns["SPY"]
spy_vol = np.sqrt(ann_cov.loc["SPY", "SPY"])
spy_sharpe = (spy_return - risk_free_rate) / spy_vol

print("\nSPY (Market Portfolio):")
print(f"  Return:     {spy_return:.2%}")
print(f"  Volatility: {spy_vol:.2%}")
print(f"  Sharpe:     {spy_sharpe:.4f}")

# ---- PLOT ---- #
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    results_df["Volatility"],
    results_df["Return"],
    c=results_df["Sharpe"],
    cmap="viridis",
    alpha=0.3,
    s=8
)
plt.colorbar(scatter, label="Sharpe Ratio")

# Capital Market Line
cml_x = np.linspace(0, results_df["Volatility"].max() * 1.1, 100)
cml_slope = (max_sharpe["Return"] - risk_free_rate) / max_sharpe["Volatility"]
cml_y = risk_free_rate + cml_slope * cml_x
ax.plot(cml_x, cml_y, color="black", linewidth=1.5, linestyle="--", label="Capital Market Line")

ax.scatter(spy_vol, spy_return, color="cyan", s=200, zorder=5, marker="o", label=f"SPY ({spy_sharpe:.2f} Sharpe)")

spy_cml_slope = (spy_return - risk_free_rate) / spy_vol
spy_cml_y = risk_free_rate + spy_cml_slope * cml_x
ax.plot(cml_x, spy_cml_y, color="cyan", linewidth=1.5, linestyle="--", label="SPY Capital Market Line")

# Key portfolios
ax.scatter(max_sharpe["Volatility"], max_sharpe["Return"], color="gold", s=200, zorder=5, label=f"Max Sharpe ({max_sharpe['Sharpe']:.2f})")
ax.scatter(min_vol["Volatility"], min_vol["Return"], color="lime", s=200, zorder=5, label=f"Min Variance ({min_vol['Volatility']:.2%} vol)")
ax.scatter(your_vol, your_return, color="tomato", s=200, zorder=5, marker="*", label=f"Equal Weight ({your_sharpe:.2f} Sharpe)")

# Risk free rate
ax.axhline(y=risk_free_rate, color="grey", linestyle=":", linewidth=1, label=f"Risk Free Rate ({risk_free_rate:.2%})")

ax.set_title("Efficient Frontier (2015-2025)")
ax.set_xlabel("Annualised Volatility")
ax.set_ylabel("Annualised Return")
ax.legend(loc="upper left")
plt.savefig("efficient_frontier.png", bbox_inches="tight")
print("\nChart saved!")