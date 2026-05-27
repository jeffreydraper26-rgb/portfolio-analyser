import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---- SETTINGS ---- #
ticker1 = "SPY"
ticker2 = "QQQ"
period = "5y"

# ---- DOWNLOAD DATA ---- #
data1 = yf.download(ticker1, period=period)
data2 = yf.download(ticker2, period=period)

# ---- CALCULATE DAILY RETURNS ---- #
data1["Daily Return"] = data1["Close"].pct_change()
data2["Daily Return"] = data2["Close"].pct_change()

returns1 = data1["Daily Return"].dropna()
returns2 = data2["Daily Return"].dropna()

# ---- PLOT DISTRIBUTIONS ---- #
fig, ax = plt.subplots(figsize=(12, 6))

for returns, color, label in [
    (returns1, "steelblue", ticker1),
    (returns2, "tomato", ticker2)
]:
    ax.hist(returns, bins=100, color=color, label=label, 
            alpha=0.5, edgecolor="none", density=True, weights=np.ones(len(returns)) / len(returns) * 100)

ax.set_title("Daily Return Distributions")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Probability (%)")
ax.legend()
plt.savefig("distributions.png", bbox_inches="tight")
print("Distribution chart saved!")

# ---- BEST AND WORST DAYS TABLE ---- #
for returns, label in [(returns1, ticker1), (returns2, ticker2)]:
    print(f"\n{label} — 5 Best Days:")
    print(returns.nlargest(5).map(lambda x: f"{x:.2%}").to_string())
    print(f"\n{label} — 5 Worst Days:")
    print(returns.nsmallest(5).map(lambda x: f"{x:.2%}").to_string())