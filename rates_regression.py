import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from fredapi import Fred
import yfinance as yf

# ---- SETUP ---- #
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ---- DOWNLOAD DATA ---- #
spy = yf.download("SPY", start="2000-01-01", end="2025-01-01")
treasury = fred.get_series("DGS10", observation_start="2000-01-01", observation_end="2025-01-01")

# ---- CALCULATE SPY MONTHLY RETURNS ---- #
spy["Return"] = spy["Close"].pct_change()
spy_monthly = spy["Return"].resample("ME").sum()

# ---- CALCULATE MONTHLY CHANGE IN YIELD ---- #
treasury_monthly = treasury.resample("ME").last().diff()

# ---- COMBINE INTO ONE TABLE ---- #
combined = pd.DataFrame({
    "SPY": spy_monthly,
    "Treasury": treasury_monthly
}).dropna()

# ---- RUN REGRESSION ---- #
slope, intercept, r_value, p_value, std_err = stats.linregress(combined["Treasury"], combined["SPY"])

# ---- PRINT RESULTS ---- #
print(f"Beta:        {slope:.4f}")
print(f"Alpha:       {intercept:.6f}")
print(f"R-squared:   {r_value**2:.4f}")
print(f"P-value:     {p_value:.6f}")

# ---- PLOT ---- #
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(combined["Treasury"], combined["SPY"], alpha=0.3, color="steelblue", s=10)
ax.plot(combined["Treasury"], slope * combined["Treasury"] + intercept, color="tomato", linewidth=2, label="Regression Line")
ax.set_title("SPY Monthly Returns vs Change in 10Y Treasury Yield")
ax.set_xlabel("Monthly Change in 10Y Yield")
ax.set_ylabel("SPY Monthly Return")
ax.legend()
plt.savefig("rates_regression_monthly.png", bbox_inches="tight")
print("Chart saved!")