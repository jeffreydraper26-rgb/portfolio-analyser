import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from fredapi import Fred
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

# ---- SETUP ---- #
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ---- DOWNLOAD DATA ---- #
jpm = yf.download("JPM", start="2000-01-01", end="2025-01-01")
spy = yf.download("SPY", start="2000-01-01", end="2025-01-01")
yield_curve = fred.get_series("T10Y2Y", observation_start="2000-01-01", observation_end="2025-01-01")
credit_spreads = fred.get_series("BAMLH0A0HYM2", observation_start="2000-01-01", observation_end="2025-01-01")

# ---- CALCULATE MONTHLY RETURNS ---- #
jpm["Return"] = jpm["Close"].pct_change()
spy["Return"] = spy["Close"].pct_change()
jpm_monthly = jpm["Return"].resample("ME").sum()
spy_monthly = spy["Return"].resample("ME").sum()

# ---- RESAMPLE RATES TO MONTHLY CHANGES ---- #
yield_curve_monthly = yield_curve.resample("ME").last().diff()
credit_spreads_monthly = credit_spreads.resample("ME").last().diff()

# ---- COMBINE INTO ONE TABLE ---- #
combined = pd.DataFrame({
    "JPM": jpm_monthly,
    "SPY": spy_monthly,
    "YieldCurve": yield_curve_monthly,
    "CreditSpreads": credit_spreads_monthly
}).dropna()

# ---- RUN MULTIPLE REGRESSION ---- #
X = combined[["SPY", "YieldCurve", "CreditSpreads"]]
y = combined["JPM"]

model = LinearRegression()
model.fit(X, y)

# ---- CALCULATE R-SQUARED AND P-VALUES ---- #
r_squared = model.score(X, y)
n = len(y)
k = X.shape[1]

# Calculate p-values manually
predictions = model.predict(X)
residuals = y - predictions
residual_std = np.sqrt(np.sum(residuals**2) / (n - k - 1))
se = residual_std * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
t_stats = model.coef_ / se
p_values = [2 * (1 - stats.t.cdf(abs(t), df=n-k-1)) for t in t_stats]

# ---- PRINT RESULTS ---- #
print(f"\nOverall R-squared: {r_squared:.4f}")
print(f"Alpha (intercept): {model.intercept_:.6f}")
print(f"\n{'Factor':<20} {'Beta':>10} {'P-value':>10}")
print("-" * 42)
for factor, beta, pval in zip(X.columns, model.coef_, p_values):
    print(f"{factor:<20} {beta:>10.4f} {pval:>10.6f}")

# ---- PLOT ACTUAL VS PREDICTED ---- #
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(combined.index, y.values, color="steelblue", label="Actual JPM Returns", alpha=0.7)
ax.plot(combined.index, predictions, color="tomato", label="Model Predicted Returns", alpha=0.7)
ax.set_title("JPM Actual vs Predicted Monthly Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Monthly Return")
ax.legend()
plt.savefig("multiple_regression.png", bbox_inches="tight")
print("\nChart saved!")