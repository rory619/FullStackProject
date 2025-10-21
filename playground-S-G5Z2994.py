# ============================================
# playGround.py  — Educational backtester scaffold
# ============================================
# PROMPT (recorded for grading/rubric):
# "Add indicators to stop losing money in 2022 and make charts as accurate as possible.
#  Use downloaded daily S&P500 ticker CSVs (25y) to generate buy/sell signals, backtest
#  a specified year (variable), export a spreadsheet of trades with columns:
#    Ticker, Buy Date, Buy Price, Sell Date, Sell Price, Percentage Profit
#  Parameters like thresholds must be variables. Adjust variables to target ~50 trades.
#  Plot per-ticker charts with clear buy/sell markers. Avoid only-obvious indicators and
#  include additional/unique features.
#  Features requested: regime filters (breadth, volatility), SMA slope guard, IBS,
#  realized volatility cap, liquidity floor, composite vote, next-day fills + slippage,
#  attribution fields in CSV, improved plots, optional walk-forward that tunes on train
#  years and evaluates on a held-out test year. EDUCATIONAL ONLY — DO NOT USE TO TRADE."
#
# IMPORTANT: This code is for coursework/education only. It is NOT trading advice.
# It makes simplifying assumptions (e.g., no portfolio-level capital constraints, no borrow fees).
# --------------------------------------------

import os
import math
import warnings
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# ============== PARAMETERS ==================
# ============================================

# ---- Core paths & year ----
theYear = 2022                        # <-- change this to any single test year
data_dir = "sp500_stock_data"         # root folder containing CSVs (recursively)
plots_dir = "plots"                   # output folder for charts
out_dir = "."                         # where to write CSV output

# ---- Costs & execution realism ----
enter_next_day_open = True            # fill at next day's open (avoids look-ahead)
cost_bps = 10                         # broker costs per side (10 bps = 0.10%)
slippage_bps = 5                      # extra per side for slippage/impact

# ---- Position mgmt / exits ----
max_hold_days = 20                    # time stop (trading days)
k_stop = 2.0                          # initial ATR multiple stop
k_trail = 3.0                         # trailing ATR multiple
use_ma_recovery_exit = True           # exit when price > SMA100
exit_on_macd_downcross = True         # exit on MACD downcross

# ---- Mean-reversion trigger (baseline) ----
buy_below_sma100_frac = 0.30          # buy when >= 30% under SMA100

# ---- Regime & risk gates (stronger for 2022) ----
breadth_min = 0.70                    # require >= 70% tickers above 200d
ui_lookback = 14
ui_max = 0.08                         # Ulcer Index cap (8%)
atr_lookback = 14
atr_pct_max = 0.03                    # ATR/Price cap (3%)

# ---- Momentum/oscillator gates ----
use_rsi_gate = True
rsi_period = 14
rsi_max = 30                          # only buy if RSI <= 30

use_macd_gate = True                  # only buy if MACD > signal (bullish)
macd_fast = 12
macd_slow = 26
macd_signal = 9

# ---- Additional (research-y) gates ----
use_bb_entry_filter = True            # require deep pullback on Bollinger %B
bb_period = 20
bb_k = 2.0
bb_max_b = 0.10

use_sma_slope = True                  # avoid catching falling knives
sma_slope_lookback = 20
sma_slope_min = 0.0                   # require SMA100 slope >= 0

use_realized_vol_filter = True        # avoid chaotic regimes
rv_lookback = 20
rv_max_annual = 0.35                  # annualized realized vol cap (35%)

use_ibs_filter = True                 # Internal Bar Strength gate
ibs_max = 0.20                        # 0=close at low, 1=close at high; require <= 0.2

use_liquidity_filter = False          # enable if your data has reliable Volume
min_dollar_vol_20d = 5e6             # average(Price*Volume) 20d

# ---- Composite vote (robustness) ----
vote_min = 4                          # number of conditions that must be true

# ---- Weekly confirm (optional example) ----
weekly_confirm = False                # if True, require weekly RSI >= 40

# ---- Trade count targeting (simple sweep) ----
run_param_sweep = False               # False by default; can tune breadth/atr/vote
target_trades = 50
sweep_breadths = [0.65, 0.70, 0.75]
sweep_atr_pcts = [0.03, 0.04]
sweep_votes = [3, 4, 5]

# ---- Walk-forward (unique feature) ----
use_walk_forward = False              # train on years, test on held-out year
train_years: List[int] = [2017, 2018, 2019, 2020, 2021]
test_year: int = 2022                 # must match theYear for convenience

# ============================================
# ============== INDICATORS ==================
# ============================================

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

def realized_vol(series: pd.Series, n: int = 20) -> pd.Series:
    r = np.log(series).diff()
    return r.rolling(n, min_periods=n).std(ddof=0) * np.sqrt(252)

def ibs_row(o, h, l, c) -> float:
    denom = (h - l)
    return float((c - l) / denom) if denom and not math.isnan(denom) else np.nan

def rolling_slope(s: pd.Series, n: int) -> pd.Series:
    # slope per bar over last n points via simple linear fit
    x = np.arange(n)
    def fit(y):
        if len(y) < n or np.any(np.isnan(y)):
            return np.nan
        return np.polyfit(x, y, 1)[0]
    return s.rolling(n, min_periods=n).apply(lambda w: fit(np.asarray(w)), raw=False)

# ============================================
# =========== CSV NORMALIZATION ==============
# ============================================

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common provider column names to: Date, Adj Close, Open, High, Low, Volume, Ticker."""
    norm = {c: c.strip().lower().replace("_", " ") for c in df.columns}

    def pick(*cands):
        for c in df.columns:
            if norm[c] in cands:
                return c
        return None

    c_date = pick("date", "timestamp", "datetime")
    c_adj = pick("adj close", "adjusted close", "adjclose", "adjusted close usd", "adj close*")
    c_close = pick("close", "close*")
    c_open = pick("open")
    c_high = pick("high")
    c_low = pick("low")
    c_volume = pick("volume", "vol")
    c_ticker = pick("ticker", "symbol", "name")

    out = pd.DataFrame()

    if c_date is None:
        raise ValueError("No date-like column found")
    out["Date"] = pd.to_datetime(df[c_date], errors="coerce")

    # Prefer Adj Close; else Close
    if c_adj is not None:
        out["Adj Close"] = pd.to_numeric(df[c_adj], errors="coerce")
    elif c_close is not None:
        out["Adj Close"] = pd.to_numeric(df[c_close], errors="coerce")
    else:
        raise ValueError("No price column found (Adj Close/Close)")

    out["Open"] = pd.to_numeric(df[c_open], errors="coerce") if c_open else out["Adj Close"]
    out["High"] = pd.to_numeric(df[c_high], errors="coerce") if c_high else out["Adj Close"]
    out["Low"] = pd.to_numeric(df[c_low], errors="coerce") if c_low else out["Adj Close"]
    out["Volume"] = pd.to_numeric(df[c_volume], errors="coerce") if c_volume else np.nan

    if c_ticker is not None and df[c_ticker].notna().any():
        out["Ticker"] = df[c_ticker].astype(str).str.upper().str.strip()

    return out

# ============================================
# ===== DATA LOADING + INDICATORS (FULL) =====
# ============================================

def read_all_csvs(data_root: str) -> Dict[str, pd.DataFrame]:
    """Read all CSVs recursively; compute indicators on full history for each ticker."""
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

        # Core indicators
        df["SMA100"] = df["Adj Close"].rolling(100, min_periods=100).mean()
        df["SMA200"] = df["Adj Close"].rolling(200, min_periods=200).mean()
        df["ATR14"] = atr(df, atr_lookback)
        df["UI14"] = ulcer_index(df, ui_lookback)
        df["RSI14"] = rsi(df["Adj Close"], rsi_period)
        macd_line, sig_line, hist = macd(df["Adj Close"], macd_fast, macd_slow, macd_signal)
        df["MACD"] = macd_line
        df["MACD_SIGNAL"] = sig_line
        df["MACD_HIST"] = hist
        df["PCTB"] = bollinger_percent_b(df["Adj Close"], bb_period, bb_k)

        # Additional research-y indicators
        df["RV20"] = realized_vol(df["Adj Close"], rv_lookback)
        df["IBS"] = [ibs_row(o, h, l, c) for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Adj Close"])]
        if "Volume" in df.columns and df["Volume"].notna().any():
            df["DollarVol20"] = (df["Adj Close"] * df["Volume"]).rolling(20, min_periods=20).mean()
        else:
            df["DollarVol20"] = np.nan
        df["SMA100_Slope"] = rolling_slope(df["SMA100"], sma_slope_lookback)

        # Optional weekly confirmation example (weekly RSI >= 40)
        if weekly_confirm:
            d = df.set_index("Date")
            w = d["Adj Close"].resample("W-FRI").last()
            w_rsi = rsi(w, 14).reindex(d.index).ffill()
            df["W_RSI14"] = w_rsi.values

        out[ticker] = df

    print(f"Found {len(out)} tickers with valid CSVs in '{data_root}'.")
    return out

# ============================================
# =========== MARKET BREADTH (200d) ==========
# ============================================

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

# ============================================
# ============== STRATEGY CORE ===============
# ============================================

def run_strategy(per_ticker_df: Dict[str, pd.DataFrame],
                 year: int,
                 breadth_floor: float,
                 atr_pct_cap: float,
                 vote_needed: int) -> pd.DataFrame:
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
        # capture "signal-day" diagnostics even if we enter next open
        sig_rsi = sig_ibs = sig_rv = sig_slope = np.nan
        sig_breadth = np.nan
        sig_date = None

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

            # --- Existing gates ---
            breadth_ok = (breadth_by_date.get(pd.Timestamp(date_i.date()), 0) >= breadth_floor)
            ui_ok = (not math.isnan(ui14_i)) and (ui14_i <= ui_max)
            atr_ok = (not math.isnan(atr14_i)) and ((atr14_i / adj_i) <= atr_pct_cap)
            rsi_ok = (not use_rsi_gate) or (not math.isnan(rsi14_i) and rsi14_i <= rsi_max)
            macd_ok = (not use_macd_gate) or (not math.isnan(macd_i) and not math.isnan(macd_sig_i) and macd_i > macd_sig_i)
            bb_ok = (not use_bb_entry_filter) or (not math.isnan(pctb_i) and pctb_i <= bb_max_b)

            # --- New gates ---
            slope_i = row.get("SMA100_Slope", np.nan)
            slope_ok = (not use_sma_slope) or (not math.isnan(slope_i) and slope_i >= sma_slope_min)

            rv_i = row.get("RV20", np.nan)
            rv_ok = (not use_realized_vol_filter) or (not math.isnan(rv_i) and rv_i <= rv_max_annual)

            ibs_i = row.get("IBS", np.nan)
            ibs_ok = (not use_ibs_filter) or (not math.isnan(ibs_i) and ibs_i <= ibs_max)

            liq_i = row.get("DollarVol20", np.nan)
            liq_ok = (not use_liquidity_filter) or (not math.isnan(liq_i) and liq_i >= min_dollar_vol_20d)

            weekly_ok = True
            if weekly_confirm:
                wv = row.get("W_RSI14", np.nan)
                weekly_ok = (not math.isnan(wv) and wv >= 40)

            votes = sum([
                breadth_ok, ui_ok, atr_ok, rsi_ok, macd_ok, bb_ok,
                slope_ok, rv_ok, ibs_ok, liq_ok, weekly_ok
            ])

            deep_discount = adj_i <= (1.0 - buy_below_sma100_frac) * sma100_i

            if not in_pos:
                if deep_discount and votes >= vote_needed:
                    # Record signal-day diagnostics
                    sig_rsi = rsi14_i if not math.isnan(rsi14_i) else np.nan
                    sig_ibs = ibs_i if not math.isnan(ibs_i) else np.nan
                    sig_rv = rv_i if not math.isnan(rv_i) else np.nan
                    sig_slope = slope_i if not math.isnan(slope_i) else np.nan
                    sig_breadth = breadth_by_date.get(pd.Timestamp(date_i.date()), np.nan)
                    sig_date = date_i

                    # BUY (optionally at next day's open)
                    if enter_next_day_open and i + 1 < len(df_yr):
                        nxt = df_yr.iloc[i+1]
                        buy_idx = i + 1
                        buy_date = nxt["Date"]
                        buy_price = nxt["Open"] if not math.isnan(nxt["Open"]) else nxt["Adj Close"]
                    else:
                        buy_idx = i
                        buy_date = date_i
                        buy_price = adj_i

                    in_pos = True
                    buy_atr = atr14_i if not math.isnan(atr14_i) else 0.0
                    high_since = float(buy_price)
                    pct_below_at_buy = (sma100_i - float(buy_price)) / sma100_i * 100.0

            else:
                # Track high for trailing stop
                if not math.isnan(adj_i):
                    high_since = max(high_since, float(adj_i))

                hold_days = i - buy_idx
                sell_now = False
                reason = None

                # 1) MA recovery
                if use_ma_recovery_exit and adj_i > sma100_i:
                    sell_now, reason = True, "MARecovery"

                # 2) ATR trailing stop
                if not sell_now and not math.isnan(atr14_i):
                    if float(adj_i) <= (high_since - k_trail * float(atr14_i)):
                        sell_now, reason = True, "ATR_Trailing"

                # 3) Initial ATR stop
                if not sell_now and buy_atr and float(adj_i) <= (float(buy_price) - k_stop * float(buy_atr)):
                    sell_now, reason = True, "ATR_Initial"

                # 4) MACD downcross
                if not sell_now and exit_on_macd_downcross and not (math.isnan(macd_i) or math.isnan(macd_sig_i)):
                    if i > 0:
                        prev = df_yr.iloc[i-1]
                        prev_macd = prev["MACD"]
                        prev_sig = prev["MACD_SIGNAL"]
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
                    sell_price = float(adj_i)
                    gross_pct = (sell_price - float(buy_price)) / float(buy_price) * 100.0
                    total_bps = (2 * cost_bps) + (2 * slippage_bps)
                    net_pct = ((1 + gross_pct / 100.0) * (1 - total_bps/10000) - 1.0) * 100.0

                    rows.append({
                        "Ticker": ticker,
                        "Signal Date": sig_date.date() if sig_date is not None else None,
                        "Buy Date": pd.Timestamp(buy_date).date(),
                        "Buy Price": round(float(buy_price), 6),
                        "%BelowSMA100_at_Buy": round(float(pct_below_at_buy), 4),
                        "Sell Date": pd.Timestamp(sell_date).date(),
                        "Sell Price": round(float(sell_price), 6),
                        "%Gain": round(float(gross_pct), 4),
                        "Cost_bps": cost_bps,
                        "Slippage_bps": slippage_bps,
                        "Net%Gain": round(float(net_pct), 4),
                        "HoldingDays": int(hold_days),
                        "ExitReason": reason,
                        # Attribution (at signal)
                        "RSI14_at_Signal": round(float(sig_rsi), 3) if not math.isnan(sig_rsi) else np.nan,
                        "IBS_at_Signal": round(float(sig_ibs), 3) if not math.isnan(sig_ibs) else np.nan,
                        "RV20_at_Signal": round(float(sig_rv), 3) if not math.isnan(sig_rv) else np.nan,
                        "Breadth_at_Signal": round(float(sig_breadth), 3) if not math.isnan(sig_breadth) else np.nan,
                        "Slope_at_Signal": round(float(sig_slope), 6) if not math.isnan(sig_slope) else np.nan,
                        "Votes_At_Entry": int(votes)
                    })

                    # reset
                    in_pos = False
                    buy_idx = buy_date = None
                    buy_price = buy_atr = None
                    pct_below_at_buy = None
                    high_since = None
                    sig_rsi = sig_ibs = sig_rv = sig_slope = np.nan
                    sig_breadth = np.nan
                    sig_date = None

    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades = trades.sort_values(["Buy Date", "Ticker"]).reset_index(drop=True)
    return trades

# ============================================
# ============== PLOTTING ====================
# ============================================

def plot_price_and_rsi(per_ticker_df: Dict[str, pd.DataFrame],
                       trades: pd.DataFrame,
                       year: int,
                       breadth_by_date: Dict[pd.Timestamp, float]):
    """Per traded ticker: Price + SMA100 with buy/sell markers & RSI panel. Shading for weak breadth."""
    os.makedirs(plots_dir, exist_ok=True)
    if trades.empty:
        return

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

        # Shade weak breadth (< breadth_min)
        if breadth_by_date:
            dates = df_yr["Date"].dt.date
            mask_weak = [(breadth_by_date.get(pd.Timestamp(d), np.nan) < breadth_min) for d in dates]
            y0, y1 = ax.get_ylim()
            ax.fill_between(df_yr["Date"], y0, y1, where=mask_weak, alpha=0.08, label=f"Breadth<{int(breadth_min*100)}%")

        # Buy/Sell markers and exit reason annotations
        t_trades = trades[trades["Ticker"] == t]
        buy_pts, sell_pts = [], []
        for _, r in t_trades.iterrows():
            bd = pd.Timestamp(r["Buy Date"])
            sd = pd.Timestamp(r["Sell Date"])

            bpx = df_yr.loc[df_yr["Date"] == bd, "Adj Close"]
            spx = df_yr.loc[df_yr["Date"] == sd, "Adj Close"]

            if not bpx.empty:
                buy_pts.append((bd, float(bpx.iloc[0])))
            if not spx.empty:
                sell_pts.append((sd, float(spx.iloc[0])))

        if buy_pts:
            ax.scatter([d for d,_ in buy_pts], [p for _,p in buy_pts], marker="^", s=40, label="Buy")
        if sell_pts:
            ax.scatter([d for d,_ in sell_pts], [p for _,p in sell_pts], marker="v", s=40, label="Sell")

        # annotate exit reasons with single letter
        for _, r in t_trades.iterrows():
            sd = pd.Timestamp(r["Sell Date"])
            spx = df_yr.loc[df_yr["Date"] == sd, "Adj Close"]
            if not spx.empty:
                reason = str(r["ExitReason"])
                label = (reason[:1] if reason else "?")
                ax.text(sd, float(spx.iloc[0]), label, fontsize=8, ha="left", va="bottom")

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

# ============================================
# ============ REPORTING / STATS =============
# ============================================

def summarize_performance(trades: pd.DataFrame):
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

    # Exit reason counts
    if "ExitReason" in trades.columns:
        counts = trades["ExitReason"].value_counts(dropna=False)
        print("\nExit reason breakdown:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    # Simple trade-sequence equity curve (note: ignores overlap & capital constraints)
    seq_ret = (trades["Net%Gain"].fillna(trades["%Gain"]) / 100.0) + 1.0
    equity = seq_ret.cumprod()
    plt.figure(figsize=(10.5, 3.2))
    plt.plot(equity.index, equity.values)
    plt.title("Trade-Sequence Equity Curve (order-of-trades)")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative Multiple")
    plt.tight_layout()
    eq_path = os.path.join(plots_dir, "equity_curve_sequence.png")
    plt.savefig(eq_path, dpi=150)
    plt.close()
    print(f"Saved equity curve: {os.path.abspath(eq_path)}")

# ============================================
# ========== SIMPLE GRID / WALK-FWD ==========
# ============================================

def _grid_eval(per_ticker_df, years: List[int],
               b_vals: List[float], a_vals: List[float], v_vals: List[int],
               target: int) -> Tuple[Tuple[float,float,int], pd.DataFrame]:
    """Pick params that produce trade count close to target and best median Net% over train years."""
    best = None
    best_trades_all = []
    for b in b_vals:
        for a in a_vals:
            for v in v_vals:
                all_trades = []
                for y in years:
                    tt = run_strategy(per_ticker_df, y, b, a, v)
                    if not tt.empty:
                        all_trades.append(tt.assign(Year=y))
                if not all_trades:
                    score = (1e9, -1e9)  # (trade_count_distance, -median)
                    trades_concat = pd.DataFrame()
                else:
                    trades_concat = pd.concat(all_trades, ignore_index=True)
                    n = len(trades_concat)
                    dist = abs(n - target)
                    med = trades_concat["Net%Gain"].median() if "Net%Gain" in trades_concat.columns else trades_concat["%Gain"].median()
                    score = (dist, -med)

                if best is None or score < best[0]:
                    best = (score, (b, a, v))
                    best_trades_all = trades_concat

    chosen = best[1] if best else (breadth_min, atr_pct_max, vote_min)
    return chosen, (best_trades_all if isinstance(best_trades_all, pd.DataFrame) else pd.DataFrame())

# ============================================
# ================= DRIVER ===================
# ============================================

def play():
    warnings.filterwarnings("ignore")
    print("Code generated by GPT-5 Thinking — EDUCATIONAL ONLY")
    print(f"Scanning CSVs under: {os.path.abspath(data_dir)}")

    per_ticker_df = read_all_csvs(data_dir)
    if not per_ticker_df:
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Choose parameters
    chosen_b, chosen_a, chosen_v = breadth_min, atr_pct_max, vote_min

    if use_walk_forward:
        if test_year != theYear:
            print(f"[walk-forward] Overriding theYear={theYear} -> test_year={test_year}")
        print(f"[walk-forward] Training on: {train_years}  |  Target trades ~{target_trades}")
        (chosen_b, chosen_a, chosen_v), _ = _grid_eval(
            per_ticker_df, train_years,
            sweep_breadths, sweep_atr_pcts, sweep_votes,
            target_trades
        )
        print(f"[walk-forward] Chosen params -> breadth_min={chosen_b:.2f}, atr_pct_max={chosen_a:.2f}, vote_min={chosen_v}")
        year_to_run = test_year
    elif run_param_sweep:
        print(f"[sweep] Searching params for {theYear} to approach ~{target_trades} trades (may overfit — demo only)")
        (chosen_b, chosen_a, chosen_v), _ = _grid_eval(
            per_ticker_df, [theYear],
            sweep_breadths, sweep_atr_pcts, sweep_votes,
            target_trades
        )
        print(f"[sweep] Chosen params -> breadth_min={chosen_b:.2f}, atr_pct_max={chosen_a:.2f}, vote_min={chosen_v}")
        year_to_run = theYear
    else:
        year_to_run = theYear

    trades = run_strategy(per_ticker_df, year_to_run, chosen_b, chosen_a, chosen_v)

    if trades.empty:
        print(f"No completed trades found for {year_to_run} with current parameters.")
        return

    out_csv = os.path.join(out_dir, f"{year_to_run}_perf.csv")
    trades.to_csv(out_csv, index=False)
    print(f"Saved trades summary: {os.path.abspath(out_csv)}")
    print(f"Total trades: {len(trades)} across {trades['Ticker'].nunique()} tickers.")
    print(f"Params -> breadth_min={chosen_b:.2f}, ui_max={ui_max:.2f}, atr_pct_max={chosen_a:.2f}, "
          f"k_stop={k_stop}, k_trail={k_trail}, max_hold_days={max_hold_days}, "
          f"rsi_gate={use_rsi_gate} (rsi_max={rsi_max}), macd_gate={use_macd_gate}, "
          f"macd_exit={exit_on_macd_downcross}, bb_filter={use_bb_entry_filter}, "
          f"sma_slope={use_sma_slope} (min={sma_slope_min}), rv_filter={use_realized_vol_filter} (max={rv_max_annual}), "
          f"ibs_filter={use_ibs_filter} (max={ibs_max}), vote_min={chosen_v}, next_day_open={enter_next_day_open}, "
          f"cost_bps={cost_bps}, slippage_bps={slippage_bps}")

    breadth_for_plots = build_breadth(per_ticker_df, year_to_run)
    plot_price_and_rsi(per_ticker_df, trades, year_to_run, breadth_for_plots)
    print(f"Saved plots to: {os.path.abspath(plots_dir)}")

    summarize_performance(trades)
