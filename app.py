import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred
import streamlit as st

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="Portfolio Analyser", layout="wide")
st.title("Portfolio Analyser")
st.markdown("Compare your portfolio's historical performance against the efficient frontier and a benchmark.")

# ---- SIDEBAR INPUTS ---- #
st.sidebar.header("Portfolio Inputs")

# Tickers and weights
st.sidebar.subheader("Assets & Weights")
st.sidebar.markdown("Enter each ticker and its weight (as a percentage). Weights must sum to 100.")

n_assets = st.sidebar.number_input("Number of assets", min_value=2, max_value=10, value=4, step=1)

tickers = []
weights = []
for i in range(n_assets):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ticker = st.text_input(f"Ticker {i+1}", value=["SPY", "JPM", "META", "V"][i] if i < 4 else "")
    with col2:
        weight = st.number_input(f"Weight {i+1} (%)", min_value=0.0, max_value=100.0, value=25.0 if i < 4 else 0.0, step=1.0)
    tickers.append(ticker.upper().strip())
    weights.append(weight)

weights_sum = sum(weights)
if abs(weights_sum - 100) > 0.01:
    st.sidebar.warning(f"Weights sum to {weights_sum:.1f}%. They must sum to 100%.")

# Date range
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2025-01-01"))

# Benchmark
st.sidebar.subheader("Benchmark")
benchmark = st.sidebar.text_input("Benchmark ticker", value="SPY").upper().strip()

# Number of random portfolios
n_portfolios = st.sidebar.slider("Number of random portfolios", min_value=1000, max_value=20000, value=10000, step=1000)

# Run button
run = st.sidebar.button("Run Analysis", type="primary")

# ---- MAIN PANEL ---- #
if not run:
    st.info("Configure your portfolio in the sidebar and press Run Analysis.")
    st.stop()

# Validate weights
if abs(weights_sum - 100) > 0.01:
    st.error("Please make sure your weights sum to 100% before running.")
    st.stop()

# Validate tickers
if any(t == "" for t in tickers):
    st.error("Please fill in all ticker fields.")
    st.stop()

with st.spinner("Downloading data and running optimisation..."):

    try:
        # ---- DOWNLOAD DATA ---- #
        from dotenv import load_dotenv
        import os
        load_dotenv()
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        all_tickers = list(set(tickers + [benchmark]))
        prices = yf.download(all_tickers, start=str(start_date), end=str(end_date), interval="1mo")["Close"]

        # Handle single ticker edge case
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        # Reorder to match user input
        prices = prices[tickers]

        # ---- CALCULATE RETURNS ---- #
        monthly_returns = prices.pct_change().dropna()
        ann_returns = (1 + monthly_returns.mean()) ** 12 - 1
        ann_cov = monthly_returns.cov() * 12

        # ---- RISK FREE RATE ---- #
        tbill = fred.get_series("TB3MS", observation_start=str(start_date), observation_end=str(end_date))
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

        # ---- KEY PORTFOLIOS ---- #
        max_sharpe_idx = results_df["Sharpe"].idxmax()
        min_vol_idx = results_df["Volatility"].idxmin()
        max_sharpe = results_df.loc[max_sharpe_idx]
        min_vol = results_df.loc[min_vol_idx]

        # ---- USER PORTFOLIO ---- #
        user_weights = np.array([w / 100 for w in weights])
        user_return = np.dot(user_weights, ann_returns)
        user_vol = np.sqrt(user_weights @ ann_cov @ user_weights)
        user_sharpe = (user_return - risk_free_rate) / user_vol

        # ---- BENCHMARK ---- #
        if benchmark in tickers:
            bmark_return = ann_returns[benchmark]
            bmark_vol = np.sqrt(ann_cov.loc[benchmark, benchmark])
        else:
            bmark_prices = yf.download(benchmark, start=str(start_date), end=str(end_date), interval="1mo")["Close"]
            bmark_returns = bmark_prices.pct_change().dropna()
            bmark_return = (1 + bmark_returns.mean()) ** 12 - 1
            bmark_vol = bmark_returns.std() * np.sqrt(12)

        bmark_sharpe = (bmark_return - risk_free_rate) / bmark_vol

        # ---- CML LINES ---- #
        cml_x = np.linspace(0, results_df["Volatility"].max() * 1.1, 100)
        opt_cml_slope = (max_sharpe["Return"] - risk_free_rate) / max_sharpe["Volatility"]
        opt_cml_y = risk_free_rate + opt_cml_slope * cml_x
        bmark_cml_slope = (bmark_return - risk_free_rate) / bmark_vol
        bmark_cml_y = risk_free_rate + bmark_cml_slope * cml_x

        # ---- HOVER TEXT ---- #
        hover_text = []
        for i in range(n_portfolios):
            w = weights_df.iloc[i]
            text = "<br>".join([f"{t}: {w[t]:.1%}" for t in tickers])
            hover_text.append(text)

        # ---- PLOT ---- #
        fig = go.Figure()

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

        fig.add_trace(go.Scatter(
            x=cml_x, y=opt_cml_y,
            mode="lines",
            line=dict(color="gold", width=2, dash="dash"),
            name="Optimised CML"
        ))

        fig.add_trace(go.Scatter(
            x=cml_x, y=bmark_cml_y,
            mode="lines",
            line=dict(color="cyan", width=2, dash="dash"),
            name=f"{benchmark} CML"
        ))

        fig.add_trace(go.Scatter(
            x=[max_sharpe["Volatility"]], y=[max_sharpe["Return"]],
            mode="markers",
            marker=dict(color="gold", size=14, symbol="circle"),
            name=f"Max Sharpe ({max_sharpe['Sharpe']:.2f})",
            hovertemplate=f"<b>Max Sharpe</b><br>Return: {max_sharpe['Return']:.2%}<br>Volatility: {max_sharpe['Volatility']:.2%}<br>Sharpe: {max_sharpe['Sharpe']:.2f}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=[min_vol["Volatility"]], y=[min_vol["Return"]],
            mode="markers",
            marker=dict(color="lime", size=14, symbol="circle"),
            name=f"Min Variance ({min_vol['Volatility']:.2%} vol)",
            hovertemplate=f"<b>Min Variance</b><br>Return: {min_vol['Return']:.2%}<br>Volatility: {min_vol['Volatility']:.2%}<br>Sharpe: {min_vol['Sharpe']:.2f}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=[user_vol], y=[user_return],
            mode="markers",
            marker=dict(color="tomato", size=16, symbol="star"),
            name=f"Your Portfolio ({user_sharpe:.2f} Sharpe)",
            hovertemplate=f"<b>Your Portfolio</b><br>Return: {user_return:.2%}<br>Volatility: {user_vol:.2%}<br>Sharpe: {user_sharpe:.2f}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=[bmark_vol], y=[bmark_return],
            mode="markers",
            marker=dict(color="cyan", size=14, symbol="circle"),
            name=f"{benchmark} ({bmark_sharpe:.2f} Sharpe)",
            hovertemplate=f"<b>{benchmark}</b><br>Return: {bmark_return:.2%}<br>Volatility: {bmark_vol:.2%}<br>Sharpe: {bmark_sharpe:.2f}<extra></extra>"
        ))

        fig.add_hline(
            y=risk_free_rate,
            line_dash="dot",
            line_color="grey",
            annotation_text=f"Risk Free Rate ({risk_free_rate:.2%})",
            annotation_position="right"
        )

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Annualised Volatility",
            yaxis_title="Annualised Return",
            template="plotly_dark",
            height=650,
            legend=dict(x=0.01, y=0.99)
        )

        # ---- DISPLAY ---- #
        st.plotly_chart(fig, use_container_width=True)

        # ---- SUMMARY TABLE ---- #
        st.subheader("Portfolio Summary")
        summary = pd.DataFrame({
            "Portfolio": ["Your Portfolio", f"Max Sharpe", "Min Variance", benchmark],
            "Annualised Return": [f"{user_return:.2%}", f"{max_sharpe['Return']:.2%}", f"{min_vol['Return']:.2%}", f"{bmark_return:.2%}"],
            "Annualised Volatility": [f"{user_vol:.2%}", f"{max_sharpe['Volatility']:.2%}", f"{min_vol['Volatility']:.2%}", f"{bmark_vol:.2%}"],
            "Sharpe Ratio": [f"{user_sharpe:.2f}", f"{max_sharpe['Sharpe']:.2f}", f"{min_vol['Sharpe']:.2f}", f"{bmark_sharpe:.2f}"]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.subheader("Your Portfolio Weights")
        weights_summary = pd.DataFrame({
            "Ticker": tickers,
            "Weight": [f"{w:.1%}" for w in user_weights]
        })
        st.dataframe(weights_summary, hide_index=True)

        st.caption("Note: Analysis assumes portfolio weights held constant over the selected period (buy and hold). Past performance does not guarantee future results.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")