#prompts:1 add indicators to stop losing money in 2022 what are more features I can add here to be as accurate as possible. This is the project that im doing and with the code u gave me were almost at about 40% with the 2 core features so give me ideas of what I could implement: Teams of 2 will be finalized today. If you don't have a team partner, I will allocate one this week. The objective is to practice Level 3 programming, whereby code is generated entirely by AI prompts written in English. English is now the new programming language! The goal is to create a python project that uses downloaded daily S@P500 ticker data over the past 25 years to predict when to buy and sell stocks. NNB it won't work: the best minds on the planet have been trying to do this for 30 years without significant success! NO NOT USE THIS TO INVEST OR TRADE. Use GPT to find combinations of Indicators to create the buy and sell signals. And the get GPT to create the code to: - backtest your strategy for a specific year: e.g. 2024, or whatever year I decide to specify, the year should be a parameter stored in a variable that can be easily changed in code, - other parameters stored in variables should include Indicator attributes (e.g. RSI threshold), - create a spread sheet for trades with the following columns: Ticker, Buy Date, Buy Price, Sell Date, Sell Price, Percentage Profit - adjust your variables to get a finite number of trades (e.g. 50), - plot charts of the stocks traded, clearly making buy and sell signals. Make sure you don't use the obvious indicators because this could result in several similar presentations, and penalties may be incurred if it looks like your work is not original. Record the prompts as comments at the beginning of your code. Where prompts require several iterations, try to construct a single detailed prompt that doesn't require further iterations. Each student is a team will receive identical marks unless there is a dispute whereby one student claims to have contributed more that the other, in which case I will award individual marks. This assessment accounts for 25% of your overall grade for this course. Marks will be awarded as follows: - core features (basic working spreadsheet as described above, using variables to adjust parameters): up to 20% - core features (basic charts as described above, using variables to adjust parameters): up to 20% - additional strategies (features other than indicators, requires research): up to 30% - unique features (features other than any of the above, requires innovation): up to 30% The over mark calculated above will be multiplied by a number between 0 and 1 representing your engagement.
# playGround.py
#add indicators to stop losing money in 2022 what are more features I can add here to be as accurate as possible. This is the project that im doing and with the code u gave me were almost at about 40% with the 2 core features so give me ideas of what I could implement: Teams of 2 will be finalized today. If you don't have a team partner, I will allocate one this week. The objective is to practice Level 3 programming, whereby code is generated entirely by AI prompts written in English. English is now the new programming language! The goal is to create a python project that uses downloaded daily S@P500 ticker data over the past 25 years to predict when to buy and sell stocks. NNB it won't work: the best minds on the planet have been trying to do this for 30 years without significant success! NO NOT USE THIS TO INVEST OR TRADE. Use GPT to find combinations of Indicators to create the buy and sell signals. And the get GPT to create the code to: - backtest your strategy for a specific year: e.g. 2024, or whatever year I decide to specify, the year should be a parameter stored in a variable that can be easily changed in code, - other parameters stored in variables should include Indicator attributes (e.g. RSI threshold), - create a spread sheet for trades with the following columns: Ticker, Buy Date, Buy Price, Sell Date, Sell Price, Percentage Profit - adjust your variables to get a finite number of trades (e.g. 50), - plot charts of the stocks traded, clearly making buy and sell signals. Make sure you don't use the obvious indicators because this could result in several similar presentations, and penalties may be incurred if it looks like your work is not original. Record the prompts as comments at the beginning of your code. Where prompts require several iterations, try to construct a single detailed prompt that doesn't require further iterations. Each student is a team will receive identical marks unless there is a dispute whereby one student claims to have contributed more that the other, in which case I will award individual marks. This assessment accounts for 25% of your overall grade for this course. Marks will be awarded as follows: - core features (basic working spreadsheet as described above, using variables to adjust parameters): up to 20% - core features (basic charts as described above, using variables to adjust parameters): up to 20% - additional strategies (features other than indicators, requires research): up to 30% - unique features (features other than any of the above, requires innovation): up to 30% The over mark calculated above will be multiplied by a number between 0 and 1 representing your engagement.
#This script:
#- Reads ALL CSVs recursively from `data_dir` (case-insensitive .csv), normalizes columns from most providers.
#- Computes indicators on full history: SMA100/200, ATR(14), Ulcer Index(14), RSI(14), MACD(12,26,9), Bollinger %B(20,2).
#- Builds market breadth for the target year: % of tickers above their 200-day MA.
#- ENTRY (mean-reversion core): AdjClose <= (1 - buy_below_sma100_frac) * SMA100
#  gated by: breadth >= breadth_min, UI14 <= ui_max, ATR14/price <= atr_pct_max,
#  optional RSI <= rsi_max, optional MACD > signal, optional %B <= threshold.
#- EXIT (first match after entry): MA-recovery (AdjClose > SMA100) OR ATR trailing stop OR ATR initial stop
#  OR MACD downcross OR time stop OR end of year.
#- Saves trades to `<theYear>_perf.csv`, makes per-ticker price plots + RSI plots into `plots/`,
#  prints average percentage returns and stats, and prints GPT version string.
import os
import math
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ====== PARAMETERS =======
# =========================
theYear = 2024
data_dir = "sp500_stock_data"           # <- change this if your folder name differs (e.g., "sp500_stock_data")
plots_dir = "plots"
max_hold_days = 40
cost_bps = 10                 # per side (entry+exit), e.g., 10 bps = 0.10%

# Core mean-reversion trigger
buy_below_sma100_frac = 0.30  # buy when >=30% below SMA100

# Regime / risk gates
breadth_min = 0.60            # require >= 60% of tickers above their 200d
ui_lookback = 14
ui_max = 0.08                 # Ulcer Index ceiling (0.08 = 8%)
atr_lookback = 14
atr_pct_max = 0.05            # ATR14 / price <= 5%

# Momentum / oscillator gates (optional)
use_rsi_gate = True
rsi_period = 14
rsi_max = 30                  # only buy if RSI <= 30 (oversold)
use_macd_gate = True          # only buy if MACD > signal (bullish)
macd_fast = 12
macd_slow = 26
macd_signal = 9

# Extra indicator (Bollinger %B)
use_bb_entry_filter = False   # optional: buy only if %B <= bb_max_b
bb_period = 20
bb_k = 2.0
bb_max_b = 0.10

# Exits / stops
use_ma_recovery_exit = True   # sell if AdjClose > SMA100
k_stop = 1.5                  # initial ATR stop multiple
k_trail = 2.5                 # trailing ATR stop multiple
exit_on_macd_downcross = True # sell when MACD crosses below signal

# Optional tiny sweep to target ~50 trades (off by default)
run_param_sweep = False
sweep_breadths = [0.55, 0.60, 0.65]
sweep_atr_pcts = [0.04, 0.05, 0.06]

# ==============================
# ===== HELPER INDICATORS ======
# ==============================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_percent_b(series: pd.Series, n=20, k=2.0) -> pd.Series:
    ma = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = ma + k*sd, ma - k*sd
    return (series - lower) / (upper - lower)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Adj Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def ulcer_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    px = df["Adj Close"]
    roll_max = px.rolling(n, min_periods=n).max()
    dd = (px / roll_max - 1.0) * 100.0
    return (dd.pow(2).rolling(n, min_periods=n).mean()).pow(0.5) / 100.0

# ==============================
# ===== CSV NORMALIZATION ======
# ==============================
def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common provider column names to: Date, Adj Close, Open, High, Low, Volume, Ticker."""
    # make a simple alias map on lowercase/stripped
    norm = {c: c.strip().lower().replace("_", " ") for c in df.columns}

    def pick(*cands):
        for c in df.columns:
            if norm[c] in cands:
                return c
        return None

    c_date   = pick("date", "timestamp", "datetime")
    c_adj    = pick("adj close", "adjusted close", "adjclose", "adjusted close usd", "adj close*")
    c_close  = pick("close", "close*")
    c_open   = pick("open")
    c_high   = pick("high")
    c_low    = pick("low")
    c_volume = pick("volume", "vol")
    c_ticker = pick("ticker", "symbol", "name")

    out = pd.DataFrame()
    if c_date is None:
        raise ValueError("No date-like column found")
    out["Date"] = pd.to_datetime(df[c_date], errors="coerce", utc=False)

    # Prefer Adj Close; else Close
    if c_adj is not None:
        out["Adj Close"] = pd.to_numeric(df[c_adj], errors="coerce")
    elif c_close is not None:
        out["Adj Close"] = pd.to_numeric(df[c_close], errors="coerce")
    else:
        raise ValueError("No price column found (Adj Close/Close)")

    # Optional OHLCV
    out["Open"]   = pd.to_numeric(df[c_open], errors="coerce")   if c_open   else out["Adj Close"]
    out["High"]   = pd.to_numeric(df[c_high], errors="coerce")   if c_high   else out["Adj Close"]
    out["Low"]    = pd.to_numeric(df[c_low], errors="coerce")    if c_low    else out["Adj Close"]
    out["Volume"] = pd.to_numeric(df[c_volume], errors="coerce") if c_volume else np.nan

    if c_ticker is not None and df[c_ticker].notna().any():
        out["Ticker"] = df[c_ticker].astype(str).str.upper().str.strip()

    return out

# ==============================
# ===== DATA LOADING + IND =====
# ==============================
def read_all_csvs(data_root: str) -> Dict[str, pd.DataFrame]:
    """Read all CSVs recursively; compute indicators on full history."""
    csv_paths = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, f))
    if not csv_paths:
        print(f"No CSVs found under '{data_root}'.")
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(csv_paths):
        try:
            raw = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        try:
            df = normalize_ohlcv_columns(raw)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # Ticker best-effort
        if "Ticker" in df.columns and df["Ticker"].notna().any():
            ticker = str(df["Ticker"].dropna().astype(str).mode().iloc[0]).strip().upper()
        else:
            ticker = os.path.splitext(os.path.basename(path))[0].strip().upper()

        # Indicators on full history
        df["SMA100"] = df["Adj Close"].rolling(100, min_periods=100).mean()
        df["SMA200"] = df["Adj Close"].rolling(200, min_periods=200).mean()
        df["ATR14"]  = atr(df, atr_lookback)
        df["UI14"]   = ulcer_index(df, ui_lookback)
        df["RSI14"]  = rsi(df["Adj Close"], rsi_period)
        macd_line, sig_line, hist = macd(df["Adj Close"], macd_fast, macd_slow, macd_signal)
        df["MACD"] = macd_line
        df["MACD_SIGNAL"] = sig_line
        df["MACD_HIST"] = hist
        df["PCTB"] = bollinger_percent_b(df["Adj Close"], bb_period, bb_k)

        out[ticker] = df

    print(f"Found {len(out)} tickers with valid CSVs in '{data_root}'.")
    return out

def build_breadth(per_ticker_df: Dict[str, pd.DataFrame], year: int) -> Dict[pd.Timestamp, float]:
    """% of tickers above their 200d on each business day of the year."""
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    all_dates = pd.date_range(start, end, freq="B")
    breadth: Dict[pd.Timestamp, float] = {}
    indexed = {t: df.set_index("Date") for t, df in per_ticker_df.items()}

    for d in all_dates:
        ok = total = 0
        for df in indexed.values():
            if d in df.index:
                sma200 = df.at[d, "SMA200"] if "SMA200" in df.columns else np.nan
                px = df.at[d, "Adj Close"]
                if not (np.isnan(px) or np.isnan(sma200)):
                    total += 1
                    if px > sma200:
                        ok += 1
        breadth[d] = (ok / total) if total else np.nan
    return breadth

# ==================================
# ======== BACKTEST ENGINE =========
# ==================================
def run_strategy(per_ticker_df: Dict[str, pd.DataFrame],
                 year: int,
                 breadth_floor: float,
                 atr_pct_cap: float) -> pd.DataFrame:
    breadth_by_date = build_breadth(per_ticker_df, year)
    rows = []

    for ticker, df_full in per_ticker_df.items():
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        df_yr = df_full[(df_full["Date"] >= start) & (df_full["Date"] <= end)].reset_index(drop=True)
        if df_yr.empty or df_yr["SMA100"].isna().all():
            continue

        in_pos = False
        buy_idx = None
        buy_date = None
        buy_price = None
        buy_atr = None
        pct_below_at_buy = None
        high_since = None

        for i in range(len(df_yr)):
            row = df_yr.iloc[i]
            date_i = row["Date"]
            adj_i = row["Adj Close"]
            sma100_i = row["SMA100"]
            atr14_i = row["ATR14"]
            ui14_i = row["UI14"]
            rsi14_i = row["RSI14"]
            macd_i = row["MACD"]
            macd_sig_i = row["MACD_SIGNAL"]
            pctb_i = row["PCTB"]

            if any(math.isnan(x) for x in [adj_i, sma100_i]):
                continue

            # Regime / risk gates
            breadth_ok = (breadth_by_date.get(pd.Timestamp(date_i.date()), 0) >= breadth_floor)
            ui_ok = (not math.isnan(ui14_i)) and (ui14_i <= ui_max)
            atr_ok = (not math.isnan(atr14_i)) and ((atr14_i / adj_i) <= atr_pct_cap)

            # Momentum / optional gates
            rsi_ok  = (not use_rsi_gate)  or (not math.isnan(rsi14_i) and rsi14_i <= rsi_max)
            macd_ok = (not use_macd_gate) or (not math.isnan(macd_i) and not math.isnan(macd_sig_i) and macd_i > macd_sig_i)
            bb_ok   = (not use_bb_entry_filter) or (not math.isnan(pctb_i) and pctb_i <= bb_max_b)

            if not in_pos:
                deep_discount = adj_i <= (1.0 - buy_below_sma100_frac) * sma100_i
                if deep_discount and breadth_ok and ui_ok and atr_ok and rsi_ok and macd_ok and bb_ok:
                    # BUY
                    in_pos = True
                    buy_idx = i
                    buy_date = date_i
                    buy_price = adj_i
                    buy_atr = atr14_i if not math.isnan(atr14_i) else 0.0
                    high_since = adj_i
                    pct_below_at_buy = (sma100_i - adj_i) / sma100_i * 100.0
            else:
                # Track high for trailing stop
                if not math.isnan(adj_i):
                    high_since = max(high_since, adj_i)

                hold_days = i - buy_idx
                sell_now = False
                reason = None

                # 1) MA recovery
                if use_ma_recovery_exit and adj_i > sma100_i:
                    sell_now, reason = True, "MARecovery"

                # 2) ATR trailing stop
                if not sell_now and not math.isnan(atr14_i):
                    if adj_i <= (high_since - k_trail * atr14_i):
                        sell_now, reason = True, "ATR_Trailing"

                # 3) Initial ATR stop
                if not sell_now and buy_atr and adj_i <= (buy_price - k_stop * buy_atr):
                    sell_now, reason = True, "ATR_Initial"

                # 4) MACD downcross exit
                if not sell_now and exit_on_macd_downcross and not (math.isnan(macd_i) or math.isnan(macd_sig_i)):
                    prev = df_yr.iloc[i-1] if i > 0 else None
                    if prev is not None:
                        prev_macd = prev["MACD"]
                        prev_sig  = prev["MACD_SIGNAL"]
                        if not (math.isnan(prev_macd) or math.isnan(prev_sig)):
                            if (prev_macd > prev_sig) and (macd_i <= macd_sig_i):
                                sell_now, reason = True, "MACD_DownCross"

                # 5) Time stop
                if not sell_now and hold_days > max_hold_days:
                    sell_now, reason = True, "TimeStop"

                # 6) End of year
                if not sell_now and i == len(df_yr) - 1:
                    sell_now, reason = True, "EndOfYear"

                if sell_now:
                    sell_date = date_i
                    sell_price = adj_i
                    gross_pct = (sell_price - buy_price) / buy_price * 100.0
                    total_cost = 2 * cost_bps / 100.0
                    net_pct = ((1 + gross_pct / 100.0) * (1 - total_cost) - 1.0) * 100.0

                    rows.append({
                        "Ticker": ticker,
                        "Buy Date": buy_date.date(),
                        "Buy Price": round(float(buy_price), 6),
                        "%BelowSMA100_at_Buy": round(float(pct_below_at_buy), 4),
                        "Sell Price": round(float(sell_price), 6),
                        "Sell Date": sell_date.date(),
                        "%Gain": round(float(gross_pct), 4),
                        "Cost_bps": cost_bps,
                        "Net%Gain": round(float(net_pct), 4),
                        "HoldingDays": int(hold_days),
                        "ExitReason": reason
                    })

                    # reset
                    in_pos = False
                    buy_idx = buy_date = None
                    buy_price = buy_atr = None
                    pct_below_at_buy = None
                    high_since = None

    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades = trades.sort_values("Buy Date").reset_index(drop=True)
    return trades

# ==============================
# ======= PLOTS & STATS ========
# ==============================
def plot_price_and_rsi(per_ticker_df: Dict[str, pd.DataFrame],
                       trades: pd.DataFrame,
                       year: int,
                       breadth_by_date: Dict[pd.Timestamp, float]):
    """Per traded ticker: price/SMA100 + buy/sell vlines; separate RSI plot."""
    os.makedirs(plots_dir, exist_ok=True)
    traded = trades["Ticker"].unique().tolist()

    for t in traded:
        df_full = per_ticker_df[t]
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        df_yr = df_full[(df_full["Date"] >= start) & (df_full["Date"] <= end)].reset_index(drop=True)
        if df_yr.empty or df_yr["Adj Close"].isna().all():
            continue

        # Price chart
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(df_yr["Date"], df_yr["Adj Close"], label="Adj Close")
        ax.plot(df_yr["Date"], df_yr["SMA100"], label="SMA100")

        # Shade weak breadth
        if breadth_by_date:
            dates = df_yr["Date"].dt.date
            mask_weak = [(breadth_by_date.get(pd.Timestamp(d), np.nan) < breadth_min) for d in dates]
            y0, y1 = ax.get_ylim()
            ax.fill_between(df_yr["Date"], y0, y1, where=mask_weak, alpha=0.08, label=f"Breadth<{int(breadth_min*100)}%")

        # Buy/Sell vertical dashed lines
        t_trades = trades[trades["Ticker"] == t]
        for _, r in t_trades.iterrows():
            ax.axvline(pd.Timestamp(r["Buy Date"]), linestyle="--", linewidth=1.0)
            ax.axvline(pd.Timestamp(r["Sell Date"]), linestyle="--", linewidth=1.0)

        ax.set_title(f"{t} — {year} (Price)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(os.path.join(plots_dir, f"{t}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

        # RSI chart
        if "RSI14" in df_yr.columns and not df_yr["RSI14"].isna().all():
            fig2, ax2 = plt.subplots(figsize=(11, 2.8))
            ax2.plot(df_yr["Date"], df_yr["RSI14"], label="RSI(14)")
            ax2.axhline(70, linestyle="--", linewidth=1.0)
            ax2.axhline(30, linestyle="--", linewidth=1.0)
            ax2.set_title(f"{t} — {year} (RSI)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("RSI")
            ax2.legend()
            fig2.autofmt_xdate()
            fig2.savefig(os.path.join(plots_dir, f"{t}_RSI.png"), bbox_inches="tight", dpi=150)
            plt.close(fig2)

def summarize_performance(trades: pd.DataFrame):
    """Print average percentage stats (gross & net), win-rate, avg hold."""
    if trades.empty:
        print("No trades to summarize.")
        return
    gross_avg = trades["%Gain"].mean()
    net_avg = trades["Net%Gain"].mean() if "Net%Gain" in trades.columns else np.nan
    win_rate = (trades["%Gain"] > 0).mean() * 100.0
    avg_hold = trades["HoldingDays"].mean()
    print(f"Average %Gain (gross): {gross_avg:.2f}%")
    if not math.isnan(net_avg):
        print(f"Average %Gain (net after costs): {net_avg:.2f}%")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Average holding days: {avg_hold:.1f}")

# ==============================
# ========== DRIVER ============
# ==============================
def play():
    warnings.filterwarnings("ignore")
    print("Code generated by GPT-5 Thinking")
    print(f"Scanning CSVs under: {os.path.abspath(data_dir)}")

    per_ticker_df = read_all_csvs(data_dir)
    if not per_ticker_df:
        return

    if run_param_sweep:
        best = None
        chosen = (breadth_min, atr_pct_max)
        for b in sweep_breadths:
            for a in sweep_atr_pcts:
                tt = run_strategy(per_ticker_df, theYear, b, a)
                n = len(tt)
                print(f"[sweep] breadth_min={b:.2f}, atr_pct_max={a:.2f} -> trades={n}")
                if best is None or abs(n - 50) < abs(best[0] - 50):
                    best = (n, b, a, tt)
        if best:
            n, b, a, trades = best
            print(f"[chosen] breadth_min={b:.2f}, atr_pct_max={a:.2f} -> trades={n}")
            chosen = (b, a)
    else:
        trades = run_strategy(per_ticker_df, theYear, breadth_min, atr_pct_max)
        chosen = (breadth_min, atr_pct_max)

    if trades.empty:
        print(f"No completed trades found for {theYear} with current parameters.")
        return

    out_csv = f"{theYear}_perf.csv"
    trades.to_csv(out_csv, index=False)
    print(f"Saved trades summary: {os.path.abspath(out_csv)}")
    print(f"Total trades: {len(trades)} across {trades['Ticker'].nunique()} tickers.")
    print(f"Params -> breadth_min={chosen[0]:.2f}, ui_max={ui_max:.2f}, atr_pct_max={chosen[1]:.2f}, "
          f"k_stop={k_stop}, k_trail={k_trail}, max_hold_days={max_hold_days}, "
          f"rsi_gate={use_rsi_gate} (rsi_max={rsi_max}), macd_gate={use_macd_gate}, "
          f"macd_exit={exit_on_macd_downcross}, bb_filter={use_bb_entry_filter}")

    breadth_for_plots = build_breadth(per_ticker_df, theYear)
    plot_price_and_rsi(per_ticker_df, trades, theYear, breadth_for_plots)
    print(f"Saved plots to: {os.path.abspath(plots_dir)}")

    summarize_performance(trades)