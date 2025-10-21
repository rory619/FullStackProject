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
#newest prompt: If I wanted to use different indicators to improve the accuracy of the project because I need to write python code to know exactly when to buy and sell stocks, what are the most popular trading strategies to implement to absolutely maximise the chances of making profit so I will know exactly when to buy and sell using the last 25 years of S&P500 stock data:
import os
import math
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================
# ============== PARAMETERS ==================
# ============================================

# ---- Single-year (kept for convenience) ----
theYear = 2022

# ---- Multi-year run ----
run_all_years = True
years_to_run = list(range(2014, 2026))   # 2014..2025 inclusive
results_dir = "results"

# ---- Paths ----
data_dir = "sp500_stock_data"     # CSVs under here (recursively OK)
plots_dir = "plots"               # charts output
out_dir = "."                     # single-year CSV output when not using results_dir

# ---- Strategy selector ----
# choices: "mean_reversion" | "donchian_breakout" | "keltner_pullback"
#          "crsi_pullback"  | "vol_breakout"      | "supertrend"
strategy_mode = "mean_reversion"

# ---- Costs & execution realism ----
enter_next_day_open = True            # fill at next day's open (avoids look-ahead)
cost_bps = 10                         # broker cost per side (10 bps = 0.10%)
slippage_bps = 5                      # extra per side for slippage/impact

# ---- Position mgmt / exits ----
max_hold_days = 20                    # time stop (trading days)
k_stop = 2.0                          # initial ATR multiple stop
k_trail = 3.0                         # trailing ATR multiple
use_ma_recovery_exit = True           # exit when price > SMA100
exit_on_macd_downcross = True         # exit on MACD downcross

# ---- Mean-reversion trigger (used by 'mean_reversion') ----
buy_below_sma100_frac = 0.30          # buy when >= 30% under SMA100

# ---- Regime & risk gates (tighter helps in bad years like 2022) ----
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

use_liquidity_filter = False          # enable if your CSVs have reliable Volume
min_dollar_vol_20d = 5e6             # average(Price*Volume) 20d

# ---- Composite vote (robustness) ----
vote_min = 4                          # number of regime/quality conditions that must be true

# ---- Tiny grid to target ~N trades (per-year; optional) ----
run_param_sweep = False               # off by default (can overfit)
target_trades = 50
sweep_breadths = [0.65, 0.70, 0.75]
sweep_atr_pcts = [0.03, 0.04]
sweep_votes = [3, 4, 5]

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
    x = np.arange(n)
    def fit(y):
        if len(y) < n or np.any(np.isnan(y)):
            return np.nan
        return np.polyfit(x, y, 1)[0]
    return s.rolling(n, min_periods=n).apply(lambda w: fit(np.asarray(w)), raw=False)

# --- Donchian channels ---
def donchian(df: pd.DataFrame, up_n=20, dn_n=20):
    dc_high = df["High"].rolling(up_n, min_periods=up_n).max()
    dc_low  = df["Low"].rolling(dn_n, min_periods=dn_n).min()
    return dc_high, dc_low

# --- Keltner Channels (EMA + ATR) ---
def keltner(df: pd.DataFrame, ema_n=20, atr_n=20, mult=1.5):
    mid = df["Adj Close"].ewm(span=ema_n, adjust=False, min_periods=ema_n).mean()
    tr_atr = atr(df, atr_n)
    up = mid + mult*tr_atr
    lo = mid - mult*tr_atr
    return mid, up, lo

# --- ADX (Welles Wilder) ---
def adx(df: pd.DataFrame, n=14):
    high, low, close = df["High"], df["Low"], df["Adj Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = (pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_w) * 100
    minus_di = (pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_w) * 100
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan) ) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

# --- Connors RSI (3-part average) ---
def rsi_simple(series: pd.Series, n=3):
    d = series.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0,np.nan)
    return 100 - (100/(1+rs))

def streak(series: pd.Series):
    s = np.sign(series.diff().fillna(0))
    out = []
    run = 0
    for v in s:
        run = run + 1 if v>0 else (run-1 if v<0 else 0)
        out.append(run)
    return pd.Series(out, index=series.index)

def pct_rank(series: pd.Series, n=100):
    return series.rolling(n, min_periods=n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=False)

def connors_rsi(close: pd.Series, rsi_n=3, streak_rsi_n=2, rank_n=100):
    crsi = (
        rsi_simple(close, rsi_n) +
        rsi_simple(streak(close), streak_rsi_n) +
        pct_rank(close.pct_change(), rank_n)
    ) / 3.0
    return crsi

# --- Supertrend (simple variant) ---
def supertrend(df: pd.DataFrame, atr_n=10, mult=3.0):
    hl2 = (df["High"] + df["Low"]) / 2.0
    atrn = atr(df, atr_n)
    upperband = hl2 + mult * atrn
    lowerband = hl2 - mult * atrn
    st = pd.Series(index=df.index, dtype=float)
    st.iloc[0] = upperband.iloc[0]
    for i in range(1, len(df)):
        prev = st.iloc[i-1]
        if df["Adj Close"].iloc[i] > prev:
            st.iloc[i] = max(lowerband.iloc[i], prev)
        else:
            st.iloc[i] = min(upperband.iloc[i], prev)
    return st

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

        # Research-y indicators
        df["RV20"] = realized_vol(df["Adj Close"], rv_lookback)
        df["IBS"] = [ibs_row(o, h, l, c) for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Adj Close"])]
        if "Volume" in df.columns and df["Volume"].notna().any():
            df["DollarVol20"] = (df["Adj Close"] * df["Volume"]).rolling(20, min_periods=20).mean()
        else:
            df["DollarVol20"] = np.nan
        df["SMA100_Slope"] = rolling_slope(df["SMA100"], sma_slope_lookback)

        # Strategy helpers
        df["DC55_H"], df["DC20_L"] = donchian(df, up_n=55, dn_n=20)
        df["KC_M"], df["KC_U"], df["KC_L"] = keltner(df, ema_n=20, atr_n=20, mult=1.5)
        df["ADX14"] = adx(df, 14)
        df["CRSI"] = connors_rsi(df["Adj Close"], rsi_n=3, streak_rsi_n=2, rank_n=100)
        df["SUPERT"] = supertrend(df, atr_n=10, mult=3.0)

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
        sig_vals = {}

        ema50 = df_yr["Adj Close"].ewm(span=50, adjust=False, min_periods=50).mean()
        ema50_prev = ema50.shift(1)

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

            # Regime/quality gates
            breadth_ok = (breadth_by_date.get(pd.Timestamp(date_i.date()), 0) >= breadth_floor)
            ui_ok = (not math.isnan(ui14_i)) and (ui14_i <= ui_max)
            atr_ok = (not math.isnan(atr14_i)) and ((atr14_i / adj_i) <= atr_pct_cap)
            rsi_ok = (not use_rsi_gate) or (not math.isnan(rsi14_i) and rsi14_i <= rsi_max)
            macd_ok = (not use_macd_gate) or (not math.isnan(macd_i) and not math.isnan(macd_sig_i) and macd_i > macd_sig_i)
            bb_ok = (not use_bb_entry_filter) or (not math.isnan(pctb_i) and pctb_i <= bb_max_b)

            slope_i = row.get("SMA100_Slope", np.nan)
            slope_ok = (not use_sma_slope) or (not math.isnan(slope_i) and slope_i >= sma_slope_min)

            rv_i = row.get("RV20", np.nan)
            rv_ok = (not use_realized_vol_filter) or (not math.isnan(rv_i) and rv_i <= rv_max_annual)

            ibs_i = row.get("IBS", np.nan)
            ibs_ok = (not use_ibs_filter) or (not math.isnan(ibs_i) and ibs_i <= ibs_max)

            liq_i = row.get("DollarVol20", np.nan)
            liq_ok = (not use_liquidity_filter) or (not math.isnan(liq_i) and liq_i >= min_dollar_vol_20d)

            votes = sum([breadth_ok, ui_ok, atr_ok, rsi_ok, macd_ok, bb_ok,
                         slope_ok, rv_ok, ibs_ok, liq_ok])

            # Strategy entry + possible strategy exit trigger
            exit_price_trigger = False
            if strategy_mode == "mean_reversion":
                entry_ok = adj_i <= (1.0 - buy_below_sma100_frac) * sma100_i
            elif strategy_mode == "donchian_breakout":
                entry_ok = (adj_i >= row["DC55_H"])
                exit_price_trigger = (adj_i <= row["DC20_L"])
            elif strategy_mode == "keltner_pullback":
                slope_up = (not math.isnan(ema50.iloc[i])) and (not math.isnan(ema50_prev.iloc[i])) and ((ema50.iloc[i] - ema50_prev.iloc[i]) >= 0)
                entry_ok = slope_up and (adj_i <= row["KC_L"]) and (rsi14_i <= 30)
                exit_price_trigger = (adj_i >= row["KC_M"])
            elif strategy_mode == "crsi_pullback":
                entry_ok = (row["CRSI"] <= 10) and (row.get("IBS", 0.5) <= 0.2)
                ma10 = df_yr["Adj Close"].rolling(10, min_periods=10).mean().iloc[i]
                exit_price_trigger = (adj_i >= ma10) if not math.isnan(ma10) else False
            elif strategy_mode == "vol_breakout":
                rv_ok2 = (row.get("RV20", np.nan) <= 0.30) if not math.isnan(row.get("RV20", np.nan)) else False
                entry_ok = rv_ok2 and (adj_i >= row["KC_U"]) and (row["ADX14"] >= 20)
                exit_price_trigger = (adj_i <= row["KC_M"])
            elif strategy_mode == "supertrend":
                prev_st = row["SUPERT"] if i == 0 else df_yr["SUPERT"].iloc[i-1]
                prev_px = df_yr["Adj Close"].iloc[i-1] if i > 0 else np.nan
                entry_ok = (adj_i > row["SUPERT"]) and (not math.isnan(prev_px) and prev_px <= prev_st)
                exit_price_trigger = (adj_i < row["SUPERT"])
            else:
                entry_ok = False

            can_buy = entry_ok and (votes >= vote_needed)

            if not in_pos:
                if can_buy:
                    sig_vals = {
                        "RSI14": rsi14_i, "IBS": ibs_i, "RV20": rv_i,
                        "Breadth": breadth_by_date.get(pd.Timestamp(date_i.date()), np.nan),
                        "Slope": slope_i, "CRSI": row.get("CRSI", np.nan),
                        "ADX14": row.get("ADX14", np.nan), "PCTB": pctb_i
                    }
                    # BUY (optionally next day's open)
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

                # strategy price trigger
                if exit_price_trigger:
                    sell_now, reason = True, f"{strategy_mode}_price"

                # 1) MA recovery
                if not sell_now and use_ma_recovery_exit and adj_i > sma100_i:
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
                        prev_macd = prev["MACD"]; prev_sig = prev["MACD_SIGNAL"]
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
                        "Strategy": strategy_mode,
                        "Signal Date": (pd.Timestamp(df_yr.iloc[buy_idx-1]["Date"]).date()
                                        if enter_next_day_open and buy_idx and buy_idx > 0
                                        else pd.Timestamp(buy_date).date()),
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
                        "RSI14_at_Signal": round(float(sig_vals.get("RSI14", np.nan)), 3) if not math.isnan(sig_vals.get("RSI14", np.nan)) else np.nan,
                        "IBS_at_Signal": round(float(sig_vals.get("IBS", np.nan)), 3) if not math.isnan(sig_vals.get("IBS", np.nan)) else np.nan,
                        "RV20_at_Signal": round(float(sig_vals.get("RV20", np.nan)), 3) if not math.isnan(sig_vals.get("RV20", np.nan)) else np.nan,
                        "Breadth_at_Signal": round(float(sig_vals.get("Breadth", np.nan)), 3) if not math.isnan(sig_vals.get("Breadth", np.nan)) else np.nan,
                        "Slope_at_Signal": round(float(sig_vals.get("Slope", np.nan)), 6) if not math.isnan(sig_vals.get("Slope", np.nan)) else np.nan,
                        "CRSI_at_Signal": round(float(sig_vals.get("CRSI", np.nan)), 3) if not math.isnan(sig_vals.get("CRSI", np.nan)) else np.nan,
                        "ADX14_at_Signal": round(float(sig_vals.get("ADX14", np.nan)), 3) if not math.isnan(sig_vals.get("ADX14", np.nan)) else np.nan,
                        "PCTB_at_Signal": round(float(sig_vals.get("PCTB", np.nan)), 3) if not math.isnan(sig_vals.get("PCTB", np.nan)) else np.nan,
                        "Votes_At_Entry": int(votes)
                    })

                    # reset
                    in_pos = False
                    buy_idx = buy_date = None
                    buy_price = buy_atr = None
                    pct_below_at_buy = None
                    high_since = None
                    sig_vals = {}

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

        # Price
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(df_yr["Date"], df_yr["Adj Close"], label="Adj Close")
        ax.plot(df_yr["Date"], df_yr["SMA100"], label="SMA100")

        if breadth_by_date:
            dates = df_yr["Date"].dt.date
            mask_weak = [(breadth_by_date.get(pd.Timestamp(d), np.nan) < breadth_min) for d in dates]
            y0, y1 = ax.get_ylim()
            ax.fill_between(df_yr["Date"], y0, y1, where=mask_weak, alpha=0.08, label=f"Breadth<{int(breadth_min*100)}%")

        t_trades = trades[trades["Ticker"] == t]
        buys, sells = [], []
        for _, r in t_trades.iterrows():
            bd = pd.Timestamp(r["Buy Date"]); sd = pd.Timestamp(r["Sell Date"])
            bpx = df_yr.loc[df_yr["Date"] == bd, "Adj Close"]
            spx = df_yr.loc[df_yr["Date"] == sd, "Adj Close"]
            if not bpx.empty: buys.append((bd, float(bpx.iloc[0])))
            if not spx.empty: sells.append((sd, float(spx.iloc[0])))

        if buys:  ax.scatter([d for d,_ in buys],  [p for _,p in buys],  marker="^", s=40, label="Buy")
        if sells: ax.scatter([d for d,_ in sells], [p for _,p in sells], marker="v", s=40, label="Sell")

        for _, r in t_trades.iterrows():
            sd = pd.Timestamp(r["Sell Date"])
            spx = df_yr.loc[df_yr["Date"] == sd, "Adj Close"]
            if not spx.empty:
                label = (str(r["ExitReason"])[:1] if r["ExitReason"] else "?")
                ax.text(sd, float(spx.iloc[0]), label, fontsize=8, ha="left", va="bottom")

        ax.set_title(f"{t} — {year} (Price) [{strategy_mode}]")
        ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
        fig.autofmt_xdate()
        fig.savefig(os.path.join(plots_dir, f"{t}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

        # RSI
        if "RSI14" in df_yr.columns and not df_yr["RSI14"].isna().all():
            fig2, ax2 = plt.subplots(figsize=(11, 2.8))
            ax2.plot(df_yr["Date"], df_yr["RSI14"], label="RSI(14)")
            ax2.axhline(70, linestyle="--", linewidth=1.0)
            ax2.axhline(30, linestyle="--", linewidth=1.0)
            ax2.set_title(f"{t} — {year} (RSI)")
            ax2.set_xlabel("Date"); ax2.set_ylabel("RSI"); ax2.legend()
            fig2.autofmt_xdate()
            fig2.savefig(os.path.join(plots_dir, f"{t}_RSI.png"), bbox_inches="tight", dpi=150)
            plt.close(fig2)

def plot_price_and_rsi(per_ticker_df: Dict[str, pd.DataFrame],
                       trades: pd.DataFrame,
                       year: int,
                       breadth_by_date: Dict[pd.Timestamp, float]):
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

        t_trades = trades[trades["Ticker"] == t]

        # Create figure with 3 vertical subplots (Price+SMA, Volume, RSI)
        fig, (ax_price, ax_vol, ax_rsi) = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                                      gridspec_kw={'height_ratios': [4, 1, 2]})

        # --- Price and SMA100 ---
        ax_price.plot(df_yr["Date"], df_yr["Adj Close"], label="Adj Close", color="blue")
        ax_price.plot(df_yr["Date"], df_yr["SMA100"], label="SMA100", color="orange", linestyle="--")

        # Shade weak breadth
        if breadth_by_date:
            dates = df_yr["Date"].dt.date
            mask_weak = [(breadth_by_date.get(pd.Timestamp(d), np.nan) < breadth_min) for d in dates]
            y0, y1 = ax_price.get_ylim()
            ax_price.fill_between(df_yr["Date"], y0, y1, where=mask_weak, color="red", alpha=0.1,
                                  label=f"Breadth < {int(breadth_min*100)}%")

        # Mark buy/sell points with scatter markers
        ax_price.scatter(t_trades["Buy Date"], t_trades["Buy Price"], marker="^", color="green",
                         label="Buy", s=80, zorder=5)
        ax_price.scatter(t_trades["Sell Date"], t_trades["Sell Price"], marker="v", color="red",
                         label="Sell", s=80, zorder=5)

        ax_price.set_ylabel("Price")
        ax_price.set_title(f"{t} — {year} Price + SMA100")
        ax_price.grid(True)
        ax_price.legend(loc="upper left")

        # --- Volume bars ---
        if "Volume" in df_yr.columns and not df_yr["Volume"].isna().all():
            ax_vol.bar(df_yr["Date"], df_yr["Volume"], color="gray", alpha=0.6)
            ax_vol.set_ylabel("Volume")
            ax_vol.grid(True)
        else:
            ax_vol.set_visible(False)

        # --- RSI plot ---
        if "RSI14" in df_yr.columns and not df_yr["RSI14"].isna().all():
            ax_rsi.plot(df_yr["Date"], df_yr["RSI14"], label="RSI(14)", color="purple")
            ax_rsi.axhline(70, color="red", linestyle="--", linewidth=1)
            ax_rsi.axhline(30, color="green", linestyle="--", linewidth=1)
            ax_rsi.set_ylabel("RSI")
            ax_rsi.set_xlabel("Date")
            ax_rsi.set_title("RSI (14)")
            ax_rsi.grid(True)
            ax_rsi.legend()
        else:
            ax_rsi.set_visible(False)

        # Date formatting on x-axis
        ax_rsi.xaxis.set_major_locator(mdates.MonthLocator())
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{t}_enhanced.png"), dpi=150)
        plt.close(fig)

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

    if "ExitReason" in trades.columns:
        counts = trades["ExitReason"].value_counts(dropna=False)
        print("\nExit reason breakdown:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    # Simple sequence equity curve
    seq_ret = (trades["Net%Gain"].fillna(trades["%Gain"]) / 100.0) + 1.0
    equity = seq_ret.cumprod()
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10.5, 3.2))
    plt.plot(equity.index, equity.values)
    plt.title("Trade-Sequence Equity Curve (order-of-trades)")
    plt.xlabel("Trade #"); plt.ylabel("Cumulative Multiple")
    plt.tight_layout()
    eq_path = os.path.join(plots_dir, "equity_curve_sequence.png")
    plt.savefig(eq_path, dpi=150); plt.close()
    print(f"Saved equity curve: {os.path.abspath(eq_path)}")

# ============================================
# ========== SIMPLE GRID (OPTIONAL) ==========
# ============================================

def _grid_eval_multi(per_ticker_df, years: List[int],
                     b_vals: List[float], a_vals: List[float], v_vals: List[int],
                     target: int) -> Tuple[Tuple[float,float,int], pd.DataFrame]:
    """
    Choose (breadth_min, atr_pct_max, vote_min) by training on multiple years.
    Score = closeness to target trade count + tie-breaker = higher median Net%.
    """
    best = None
    best_trades = pd.DataFrame()
    for b in b_vals:
        for a in a_vals:
            for v in v_vals:
                all_trades = []
                for y in years:
                    tt = run_strategy(per_ticker_df, y, b, a, v)
                    if not tt.empty:
                        all_trades.append(tt.assign(Year=y))
                if not all_trades:
                    dist = 1e9; med = -1e9
                    tmp = pd.DataFrame()
                else:
                    tmp = pd.concat(all_trades, ignore_index=True)
                    dist = abs(len(tmp) - target*len(years))
                    med = tmp["Net%Gain"].median() if "Net%Gain" in tmp.columns else tmp["%Gain"].median()
                score = (dist, -med)
                if best is None or score < best[0]:
                    best = (score, (b, a, v))
                    best_trades = tmp
    chosen = best[1] if best else (breadth_min, atr_pct_max, vote_min)
    return chosen, best_trades

def _choose_params_for_year(per_ticker_df, year: int):
    """Return (breadth_min, atr_pct_max, vote_min); here we just use globals, or run a tiny sweep if enabled."""
    if run_param_sweep:
        (b, a, v), _ = _grid_eval_multi(
            per_ticker_df,
            [year],  # single-year tuning (demo only; can overfit)
            sweep_breadths, sweep_atr_pcts, sweep_votes,
            target_trades
        )
        return b, a, v
    return breadth_min, atr_pct_max, vote_min

# ============================================
# ================ DRIVERS ===================
# ============================================

def run_one_year(per_ticker_df, year: int):
    """Runs one year, writes CSV to results/, plots to plots/<year>/, returns the trades DataFrame."""
    global plots_dir  # <-- must be before any reference/assignment to plots_dir

    b, a, v = _choose_params_for_year(per_ticker_df, year)
    trades = run_strategy(per_ticker_df, year, b, a, v)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"{year}_perf.csv")
    if not trades.empty:
        trades.to_csv(out_csv, index=False)
        print(f"[{year}] Saved trades: {os.path.abspath(out_csv)} (N={len(trades)})")
    else:
        print(f"[{year}] No completed trades with current parameters.")

    # plot to plots/<year> (temporarily redirect global plots_dir)
    year_plots = os.path.join(plots_dir, str(year))
    os.makedirs(year_plots, exist_ok=True)
    old_plots = plots_dir
    plots_dir = year_plots
    try:
        breadth_for_plots = build_breadth(per_ticker_df, year)
        if not trades.empty:
            plot_price_and_rsi(per_ticker_df, trades, year, breadth_for_plots)
            print(f"[{year}] Saved plots to: {os.path.abspath(year_plots)}")
        summarize_performance(trades)
    finally:
        plots_dir = old_plots


    return trades

def run_all_years_now():
    warnings.filterwarnings("ignore")
    print("Code generated by GPT-5 Thinking — EDUCATIONAL ONLY")
    print(f"Strategy: {strategy_mode}")
    print(f"Scanning CSVs under: {os.path.abspath(data_dir)}")

    per_ticker_df = read_all_csvs(data_dir)
    if not per_ticker_df:
        return

    all_trades = []
    for yr in years_to_run:
        print("\n" + "="*70)
        print(f"Running {yr} …")
        t = run_one_year(per_ticker_df, yr)
        if not t.empty:
            all_trades.append(t.assign(Year=yr))

    if all_trades:
        combo = pd.concat(all_trades, ignore_index=True)
        os.makedirs(results_dir, exist_ok=True)
        combo_csv = os.path.join(results_dir, "all_years_perf.csv")
        combo.to_csv(combo_csv, index=False)
        print(f"\nSaved all-years trades: {os.path.abspath(combo_csv)} (N={len(combo)})")

        # Per-year summary
        def win_rate_of(df):
            col = "Net%Gain" if "Net%Gain" in df.columns else "%Gain"
            return float((df[col] > 0).mean() * 100.0)

        by_year = combo.groupby("Year").apply(lambda g: pd.Series({
            "n_trades": len(g),
            "win_rate": win_rate_of(g),
            "mean_net": g["Net%Gain"].mean(),
            "median_net": g["Net%Gain"].median(),
            "mean_hold": g["HoldingDays"].mean()
        })).reset_index()

        by_year_csv = os.path.join(results_dir, "summary_by_year.csv")
        by_year.to_csv(by_year_csv, index=False)
        print(f"Saved per-year summary: {os.path.abspath(by_year_csv)}")
    else:
        print("\nNo trades across the requested years (try loosening filters or a different strategy).")

def play():
    """Legacy single-year runner."""
    warnings.filterwarnings("ignore")
    print("Code generated by GPT-5 Thinking — EDUCATIONAL ONLY")
    print(f"Strategy: {strategy_mode}")
    print(f"Scanning CSVs under: {os.path.abspath(data_dir)}")

    per_ticker_df = read_all_csvs(data_dir)
    if not per_ticker_df:
        return

    b, a, v = _choose_params_for_year(per_ticker_df, theYear)
    trades = run_strategy(per_ticker_df, theYear, b, a, v)

    if trades.empty:
        print(f"No completed trades found for {theYear} with current parameters.")
        return

    out_csv = os.path.join(out_dir, f"{theYear}_perf.csv")
    trades.to_csv(out_csv, index=False)
    print(f"Saved trades summary: {os.path.abspath(out_csv)}")
    print(f"Total trades: {len(trades)} across {trades['Ticker'].nunique()} tickers.")
    print(f"Params -> breadth_min={b:.2f}, ui_max={ui_max:.2f}, atr_pct_max={a:.2f}, "
          f"k_stop={k_stop}, k_trail={k_trail}, max_hold_days={max_hold_days}, "
          f"rsi_gate={use_rsi_gate} (rsi_max={rsi_max}), macd_gate={use_macd_gate}, "
          f"macd_exit={exit_on_macd_downcross}, bb_filter={use_bb_entry_filter}, "
          f"sma_slope={use_sma_slope} (min={sma_slope_min}), rv_filter={use_realized_vol_filter} (max={rv_max_annual}), "
          f"ibs_filter={use_ibs_filter} (max={ibs_max}), vote_min={v}, next_day_open={enter_next_day_open}, "
          f"cost_bps={cost_bps}, slippage_bps={slippage_bps}")

    breadth_for_plots = build_breadth(per_ticker_df, theYear)
    plot_price_and_rsi(per_ticker_df, trades, theYear, breadth_for_plots)
    print(f"Saved plots to: {os.path.abspath(plots_dir)}")
    summarize_performance(trades)
