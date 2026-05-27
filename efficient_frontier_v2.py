import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred

# ---- SETUP ---- #
from dotenv import load_dotenv
import os
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

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

# ---- GET RISK FREE RATE ---- #
tbill = fred.get_series("TB3MS", observation_start=start, observation_end=end)
risk_free_rate = tbill.iloc[-1] / 100

# ---- GENERATE RANDOM PORTFOLIOS ---- #
results = np.zeros((n_portfolios, 3))
weights_record = []

for i in range(n_portfolios):
    w = np.random.dirichlet(np.ones(len(tickers)) * 0.5)
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

# ---- NAMED PORTFOLIOS ---- #
your_weights = np.array([0.25, 0.25, 0.25, 0.25])
your_return = np.dot(your_weights, ann_returns)
your_vol = np.sqrt(your_weights @ ann_cov @ your_weights)
your_sharpe = (your_return - risk_free_rate) / your_vol

spy_return = ann_returns["SPY"]
spy_vol = np.sqrt(ann_cov.loc["SPY", "SPY"])
spy_sharpe = (spy_return - risk_free_rate) / spy_vol

# ---- PRINT RESULTS ---- #
print(f"\nRisk Free Rate: {risk_free_rate:.2%}")

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

print("\nYour Equal Weight Portfolio:")
print(f"  Return:     {your_return:.2%}")
print(f"  Volatility: {your_vol:.2%}")
print(f"  Sharpe:     {your_sharpe:.4f}")

print("\nSPY (Market Portfolio):")
print(f"  Return:     {spy_return:.2%}")
print(f"  Volatility: {spy_vol:.2%}")
print(f"  Sharpe:     {spy_sharpe:.4f}")

# ---- BUILD HOVER TEXT FOR RANDOM PORTFOLIOS ---- #
hover_text = []
for i in range(n_portfolios):
    w = weights_df.iloc[i]
    text = "<br>".join([f"{ticker}: {w[ticker]:.1%}" for ticker in tickers])
    hover_text.append(text)

# ---- CML LINES ---- #
cml_x = np.linspace(0, results_df["Volatility"].max() * 1.1, 100)

optimised_cml_slope = (max_sharpe["Return"] - risk_free_rate) / max_sharpe["Volatility"]
optimised_cml_y = risk_free_rate + optimised_cml_slope * cml_x

spy_cml_slope = (spy_return - risk_free_rate) / spy_vol
spy_cml_y = risk_free_rate + spy_cml_slope * cml_x

# ---- PLOT ---- #
fig = go.Figure()

# Random portfolios
fig.add_trace(go.Scatter(
    x=results_df["Volatility"],
    y=results_df["Return"],
    mode="markers",
    marker=dict(
        color=results_df["Sharpe"],
        colorscale="Viridis",
        size=4,
        opacity=0.5,
        colorbar=dict(title="Sharpe Ratio")
    ),
    text=hover_text,
    hovertemplate="<b>Return:</b> %{y:.2%}<br><b>Volatility:</b> %{x:.2%}<br><b>Weights:</b><br>%{text}<extra></extra>",
    name="Random Portfolios"
))

# Optimised CML
fig.add_trace(go.Scatter(
    x=cml_x,
    y=optimised_cml_y,
    mode="lines",
    line=dict(color="gold", width=2, dash="dash"),
    name="Optimised Capital Market Line"
))

# SPY CML
fig.add_trace(go.Scatter(
    x=cml_x,
    y=spy_cml_y,
    mode="lines",
    line=dict(color="cyan", width=2, dash="dash"),
    name="SPY Capital Market Line"
))

# Max Sharpe
fig.add_trace(go.Scatter(
    x=[max_sharpe["Volatility"]],
    y=[max_sharpe["Return"]],
    mode="markers",
    marker=dict(color="gold", size=14, symbol="circle"),
    name=f"Max Sharpe ({max_sharpe['Sharpe']:.2f})",
    hovertemplate=f"<b>Max Sharpe Portfolio</b><br>Return: {max_sharpe['Return']:.2%}<br>Volatility: {max_sharpe['Volatility']:.2%}<br>Sharpe: {max_sharpe['Sharpe']:.2f}<extra></extra>"
))

# Min Variance
fig.add_trace(go.Scatter(
    x=[min_vol["Volatility"]],
    y=[min_vol["Return"]],
    mode="markers",
    marker=dict(color="lime", size=14, symbol="circle"),
    name=f"Min Variance ({min_vol['Volatility']:.2%} vol)",
    hovertemplate=f"<b>Min Variance Portfolio</b><br>Return: {min_vol['Return']:.2%}<br>Volatility: {min_vol['Volatility']:.2%}<br>Sharpe: {min_vol['Sharpe']:.2f}<extra></extra>"
))

# Equal weight portfolio
fig.add_trace(go.Scatter(
    x=[your_vol],
    y=[your_return],
    mode="markers",
    marker=dict(color="tomato", size=14, symbol="star"),
    name=f"Equal Weight ({your_sharpe:.2f} Sharpe)",
    hovertemplate=f"<b>Your Equal Weight Portfolio</b><br>Return: {your_return:.2%}<br>Volatility: {your_vol:.2%}<br>Sharpe: {your_sharpe:.2f}<extra></extra>"
))

# SPY
fig.add_trace(go.Scatter(
    x=[spy_vol],
    y=[spy_return],
    mode="markers",
    marker=dict(color="cyan", size=14, symbol="circle"),
    name=f"SPY ({spy_sharpe:.2f} Sharpe)",
    hovertemplate=f"<b>SPY</b><br>Return: {spy_return:.2%}<br>Volatility: {spy_vol:.2%}<br>Sharpe: {spy_sharpe:.2f}<extra></extra>"
))

# Risk free rate
fig.add_hline(
    y=risk_free_rate,
    line_dash="dot",
    line_color="grey",
    annotation_text=f"Risk Free Rate ({risk_free_rate:.2%})",
    annotation_position="right"
)

fig.update_layout(
    title="Efficient Frontier (2015-2025)",
    xaxis_title="Annualised Volatility",
    yaxis_title="Annualised Return",
    template="plotly_dark",
    width=1100,
    height=700,
    legend=dict(x=0.01, y=0.99)
)

# fig.show()
fig.write_html("efficient_frontier.html")
print("\nChart saved as efficient_frontier.html")