import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------- Company List & Weights -------------------
companies = {
    "BAJFINANCE.NS": 0.20,   # Financials - very strong growth
    "TITAN.NS": 0.15,        # Consumer Discretionary - strong brand
    "ADANIENT.NS": 0.10,     # Conglomerate - solid growth
    "TCS.NS": 0.10,          # IT - global performer
    "BAJAJFINSV.NS": 0.08,   # Financials - diversified
    "NESTLEIND.NS": 0.08,    # FMCG - stable and defensive
    "NTPC.NS": 0.08,         # Energy - defensive
    "COALINDIA.NS": 0.07,    # Energy - stable cash flows
    "RELIANCE.NS": 0.07,     # Diversified - market heavyweight
    "HINDUNILVR.NS": 0.07    # FMCG - stable compounder
}

initial_cash = 1000000 # ₹10 lakhs total portfolio
START_DATE = pd.Timestamp("2018-01-01")  # backtest start (post-warmup)

# ------------------- Helpers -------------------
def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------- Indicators -------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MAs
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI(14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI14"] = 100 - (100 / (1 + rs))

    # Volume avg
    df["VOL20"] = df["Volume"].rolling(20).mean()

    # ATR(14)
    high_low = df["High"] - df["Low"]
    high_cp = (df["High"] - df["Close"].shift()).abs()
    low_cp = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # DMI/ADX(14)
    n = 14
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["up_move"] = df["High"] - df["High"].shift(1)
    df["down_move"] = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
    df["-DM"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
    df["TR_smooth"] = df["TR"].rolling(n).sum()
    df["+DM_smooth"] = df["+DM"].rolling(n).sum()
    df["-DM_smooth"] = df["-DM"].rolling(n).sum()
    df["+DI"] = 100 * (df["+DM_smooth"] / df["TR_smooth"])
    df["-DI"] = 100 * (df["-DM_smooth"] / df["TR_smooth"])
    df["DX"] = 100 * (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]))
    df["ADX14"] = df["DX"].rolling(n).mean()

    # Bollinger bandwidth & EMA10
    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BBW"] = (2 * df["BB_std"]) / df["BB_mid"]
    df["BBW126avg"] = df["BBW"].rolling(126).mean()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()

    return df

# ------------------- Strategy -------------------
def run_strategy(df: pd.DataFrame, initial_cash: float = 100_000.0, start_date: pd.Timestamp = START_DATE):
    # Clean & compute
    df = _ensure_ohlcv(df)
    df = compute_indicators(df)

    # Determine the first date when ALL needed indicators exist
    needed = ["MA20", "MA50", "RSI14", "VOL20", "ADX14", "ATR14", "BBW126avg"]
    valid_start = df[needed].dropna().index.min()
    if pd.isna(valid_start):
        # Not enough data for indicators
        return pd.DataFrame(columns=["Equity"])

    # Backtest starts at the later of (indicator-ready date, START_DATE)
    backtest_start = max(valid_start, start_date)
    df = df.loc[df.index >= backtest_start].copy()

    # Precompute previous MAs for cross checks
    prev_MA20 = df["MA20"].shift(1)
    prev_MA50 = df["MA50"].shift(1)

    position = 0
    entry_price, stop_price, size = np.nan, np.nan, 0
    cash = float(initial_cash)
    equity_curve = []
    trades = []

    for idx, row in df.iterrows():
        close = row["Close"]
        ma20, ma50 = row["MA20"], row["MA50"]
        rsi = row["RSI14"]
        vol, vol20 = row["Volume"], row["VOL20"]
        adx = row["ADX14"]
        bbw, bbw_avg = row["BBW"], row["BBW126avg"]
        atr = row["ATR14"]
        ema10 = row["EMA10"]

        # Shouldn’t happen after dropna, but keep guard
        if any(np.isnan(v) for v in (ma20, ma50, rsi, vol20, adx, bbw_avg, atr)):
            equity_curve.append(cash + (position * close if position else 0))
            continue

        if position == 0:
            cross_up = (ma20 > ma50) and (prev_MA20.loc[idx] <= prev_MA50.loc[idx])
            price_above = (close > ma20) and (close > ma50)
            rsi_ok = (rsi > 50)
            vol_ok = (vol > 1.05 * vol20)
            trend_strength = (adx > 15) or (bbw > bbw_avg)
            not_overextended = (close - ma20) < (2.5 * atr)

            if (cross_up or price_above) and rsi_ok and vol_ok and trend_strength and not_overextended:
                size = int(cash // close)  # full sleeve allocation
                if size > 0:
                    position = size
                    entry_price = close
                    stop_price = close - 1.5 * atr
                    cash -= position * close
                    trades.append({"Type": "BUY", "Date": idx, "Price": close, "Size": position})

        else:
            # trailing stop
            stop_price = max(stop_price, close - 1.5 * atr)
            cross_down = (ma20 < ma50) and (prev_MA20.loc[idx] >= prev_MA50.loc[idx])
            vol_spike = (rsi > 70) and (vol > 1.5 * vol20)
            fast_exit = close < ema10
            profit_exit = close >= entry_price + 3 * atr

            if (close <= stop_price) or cross_down or vol_spike or fast_exit or profit_exit:
                cash += position * close
                trades.append({"Type": "SELL", "Date": idx, "Price": close, "Size": position})

                position, entry_price, stop_price, size = 0, np.nan, np.nan, 0

        equity_curve.append(cash + (position * close if position else 0))

    return pd.DataFrame({"Equity": equity_curve}, index=df.index), pd.DataFrame(trades)

# ------------------- Metrics -------------------
def calculate_metrics(equity: pd.Series):
    start_value = equity.iloc[0]
    end_value = equity.iloc[-1]
    total_return = (end_value / start_value) - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan

    # Daily returns
    daily_returns = equity.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() * 252) / volatility if volatility > 0 else np.nan

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1).min()

    return {
        "Start": start_value,
        "End": end_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown,
    }

# ------------------- Portfolio Backtest -------------------
portfolio_curves = []
all_trades = {}
metrics = {}

for ticker, weight in companies.items():
    path = Path(f"nifty50_data/{ticker}.csv")
    if not path.exists():
        print(f"Data for {ticker} not found, skipping.")
        continue

    raw = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    raw = _ensure_ohlcv(raw)

    equity_df, trades_df = run_strategy(raw, initial_cash * weight, start_date=START_DATE)

    if not equity_df.empty:
        # rename column "Equity" -> ticker
        equity_df = equity_df.rename(columns={"Equity": ticker})
        portfolio_curves.append(equity_df)

        # calculate metrics on the renamed series
        metrics[ticker] = calculate_metrics(equity_df[ticker])

        all_trades[ticker] = trades_df  # store trades for each company
    else:
        print(f"{ticker}: insufficient data after indicator warm-up; skipping.")

if not portfolio_curves:
    raise RuntimeError("No equity curves generated. Check data and dates.")

# Combine all companies
portfolio = pd.concat(portfolio_curves, axis=1).sort_index()
portfolio = portfolio.fillna(method="ffill").fillna(method="bfill")

# Total portfolio = sum across tickers
portfolio["Total"] = portfolio.sum(axis=1)

# Portfolio metrics
metrics["Portfolio"] = calculate_metrics(portfolio["Total"])

# ------------------- Results -------------------
metrics_df = pd.DataFrame(metrics).T
print(metrics_df)
metrics_df.to_csv('Results/Metrics_10companies.csv')

print("\n=== TRADES PER COMPANY (sample) ===")
for ticker, tdf in all_trades.items():
    print(f"\n{ticker} Trades:")
    print(tdf.head())  # show first few trades
    tdf.to_csv(f"Results/{ticker}_Trades.csv")


# ------------------- Plot -------------------
plt.figure(figsize=(12, 6))
plt.plot(portfolio.index, portfolio["Total"], label="Portfolio")
plt.title("Top 10 Nifty50 Portfolio - Balanced Hybrid Strategy")
plt.xlabel("Date"); plt.ylabel("Portfolio Value (₹)")
plt.legend(); plt.grid(True, linestyle=":")
plt.tight_layout(); plt.show()

# ------------------- Plot with Buy/Sell -------------------
for ticker, trades_df in all_trades.items():
    path = Path(f"nifty50_data/{ticker}.csv")
    raw = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")

    plt.figure(figsize=(12, 6))
    plt.plot(raw.index, raw["Close"], label="Price")

    buys = trades_df[trades_df["Type"] == "BUY"]
    sells = trades_df[trades_df["Type"] == "SELL"]
    plt.scatter(buys["Date"], buys["Price"], marker="^", color="green", label="BUY", alpha=0.8)
    plt.scatter(sells["Date"], sells["Price"], marker="v", color="red", label="SELL", alpha=0.8)

    plt.title(f"{ticker} - Buy/Sell Signals")
    plt.xlabel("Date");
    plt.ylabel("Price")
    plt.legend();
    plt.grid(True, linestyle=":")
    plt.tight_layout();
    plt.show()
