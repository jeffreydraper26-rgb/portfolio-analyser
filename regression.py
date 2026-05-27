import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---- DOWNLOAD DATA ---- #
jpm = yf.download("JPM", period="5y")
spy = yf.download("SPY", period="5y")

# ---- CALCULATE DAILY RETURNS ---- #
jpm["Return"] = jpm["Close"].pct_change()
spy["Return"] = spy["Close"].pct_change()

# ---- COMBINE INTO ONE TABLE ---- #
combined = pd.DataFrame({
    "JPM": jpm["Return"],
    "SPY": spy["Return"]
}).dropna()

# ---- RUN REGRESSION ---- #
slope, intercept, r_value, p_value, std_err = stats.linregress(combined["SPY"], combined["JPM"])

# ---- PRINT RESULTS ---- #
print(f"Beta:        {slope:.4f}")
print(f"Alpha:       {intercept:.6f}")
print(f"R-squared:   {r_value**2:.4f}")
print(f"P-value:     {p_value:.6f}")

# ---- PLOT ---- #
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(combined["SPY"], combined["JPM"], alpha=0.3, color="steelblue", s=10)
ax.plot(combined["SPY"], slope * combined["SPY"] + intercept, color="tomato", linewidth=2, label="Regression Line")
ax.set_title("JPM vs SPY Daily Returns")
ax.set_xlabel("SPY Daily Return")
ax.set_ylabel("JPM Daily Return")
ax.legend()
plt.savefig("jpm_regression.png", bbox_inches="tight")
print("Chart saved!")