import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime, date
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WATCHLIST_TOP_N  = int(os.getenv("WATCHLIST_TOP_N", 5))

# ─── IDX TICKERS ──────────────────────────────────────────────────────────────
from tickers import IDX_TICKERS

def get_tickers():
    """Return list of IDX tickers with .JK suffix for yfinance."""
    return [f"{t}.JK" for t in IDX_TICKERS]

# ─── FETCH DATA ───────────────────────────────────────────────────────────────
def fetch_all_data(tickers):
    """Batch fetch 60 days OHLCV for all tickers."""
    print(f"📥 Fetching {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        period="30d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        multi_level_index=True,
    )
    return raw

def extract_ticker_data(raw, tickers):
    """Extract per-ticker DataFrame from batch download result."""
    result = {}
    for ticker in tickers:
        try:
            # yfinance 1.x returns MultiIndex (ticker, field)
            if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
                # MultiIndex: swap to (field, ticker) then access ticker
                df = raw.xs(ticker, axis=1, level=0) if ticker in raw.columns.get_level_values(0) else None
                if df is None:
                    df = raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) else None
            else:
                df = raw[ticker].copy() if ticker in raw.columns else None
            if df is None:
                continue
            df = df.copy()
            df = df.dropna(subset=["Close"])
            if len(df) >= 10:
                result[ticker] = df
        except Exception:
            pass
    print(f"✅ {len(result)} tickers berhasil diproses")
    return result

# ─── MARKET OVERVIEW ──────────────────────────────────────────────────────────
def get_ihsg():
    """Fetch IHSG data."""
    try:
        df = yf.download("^JKSE", period="5d", interval="1d", progress=False, auto_adjust=True)
        df = df.dropna()
        close_today = float(df["Close"].iloc[-1])
        close_prev  = float(df["Close"].iloc[-2])
        change_pct  = (close_today - close_prev) / close_prev * 100
        return close_today, change_pct
    except Exception as e:
        return None, None

def get_global_data():
    """Fetch global indices, commodities, USD/IDR."""
    symbols = {
        "S&P500":     "^GSPC",
        "Nasdaq":     "^IXIC",
        "DJIA":       "^DJI",
        "USD/IDR":    "IDR=X",
        "Emas":       "GC=F",
        "Crude Oil":  "CL=F",
        "Nikel":      "NTR.TO",  # Nickel ETF as proxy
        "US10Y":      "^TNX",
    }
    result = {}
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=True)
            df = df.dropna()
            c  = float(df["Close"].iloc[-1])
            p  = float(df["Close"].iloc[-2])
            pct = (c - p) / p * 100
            result[name] = {"value": c, "pct": pct}
        except Exception:
            result[name] = None
    return result

# ─── MOVERS ───────────────────────────────────────────────────────────────────
def compute_movers(ticker_data):
    """Compute daily change % and volume for all tickers."""
    rows = []
    for ticker, df in ticker_data.items():
        if len(df) < 2:
            continue
        try:
            close_today = float(df["Close"].iloc[-1])
            close_prev  = float(df["Close"].iloc[-2])
            volume      = float(df["Volume"].iloc[-1])
            vol_avg30   = float(df["Volume"].tail(31).iloc[:-1].mean())
            high        = float(df["High"].iloc[-1])
            low         = float(df["Low"].iloc[-1])
            open_       = float(df["Open"].iloc[-1])
            close_prev2 = float(df["Close"].iloc[-2])

            change_pct  = (close_today - close_prev) / close_prev * 100
            vol_ratio   = volume / vol_avg30 if vol_avg30 > 0 else 0
            gap_pct     = (open_ - close_prev2) / close_prev2 * 100

            # High/low close ratio (0-1)
            day_range   = high - low
            hcr = (close_today - low) / day_range if day_range > 0 else 0.5

            # 52 week high/low
            w52_high = float(df["Close"].tail(252).max())
            w52_low  = float(df["Close"].tail(252).min())
            pct_from_ath = (close_today - w52_high) / w52_high * 100
            pct_from_atl = (close_today - w52_low)  / w52_low  * 100

            rows.append({
                "ticker":       ticker.replace(".JK", ""),
                "close":        close_today,
                "change_pct":   change_pct,
                "volume":       volume,
                "vol_ratio":    vol_ratio,
                "gap_pct":      gap_pct,
                "hcr":          hcr,
                "pct_from_ath": pct_from_ath,
                "pct_from_atl": pct_from_atl,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

# ─── TECHNICAL SIGNALS ────────────────────────────────────────────────────────
def compute_technicals(ticker_data):
    """Compute RSI, MA, MACD, BB, ATR per ticker."""
    signals = []
    for ticker, df in ticker_data.items():
        if len(df) < 30:
            continue
        try:
            close = df["Close"].squeeze()
            high  = df["High"].squeeze()
            low   = df["Low"].squeeze()
            vol   = df["Volume"].squeeze()

            # RSI
            rsi_series = ta.rsi(close, length=14)
            rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None

            # MA
            ma20  = float(ta.sma(close, length=20).iloc[-1])
            ma50  = float(ta.sma(close, length=50).iloc[-1]) if len(df) >= 50 else None
            ma200 = float(ta.sma(close, length=200).iloc[-1]) if len(df) >= 200 else None

            # MACD
            macd_df = ta.macd(close)
            macd_line   = float(macd_df["MACD_12_26_9"].iloc[-1])   if macd_df is not None else None
            signal_line = float(macd_df["MACDs_12_26_9"].iloc[-1])  if macd_df is not None else None
            macd_prev   = float(macd_df["MACD_12_26_9"].iloc[-2])   if macd_df is not None else None
            signal_prev = float(macd_df["MACDs_12_26_9"].iloc[-2])  if macd_df is not None else None

            # Bollinger Bands
            bb = ta.bbands(close, length=20)
            bb_upper = float(bb["BBU_20_2.0"].iloc[-1]) if bb is not None else None
            bb_lower = float(bb["BBL_20_2.0"].iloc[-1]) if bb is not None else None
            bb_width = float(bb["BBB_20_2.0"].iloc[-1]) if bb is not None else None
            bb_width_prev = float(bb["BBB_20_2.0"].iloc[-6]) if bb is not None and len(bb) >= 6 else None

            # ATR
            atr = ta.atr(high, low, close, length=14)
            atr_val = float(atr.iloc[-1]) if atr is not None else None

            close_today = float(close.iloc[-1])
            close_prev  = float(close.iloc[-2])

            # Golden / Death cross
            golden_cross = death_cross = False
            if ma50 and ma200:
                ma50_prev  = float(ta.sma(close, length=50).iloc[-2])
                ma200_prev = float(ta.sma(close, length=200).iloc[-2])
                golden_cross = ma50_prev <= ma200_prev and ma50 > ma200
                death_cross  = ma50_prev >= ma200_prev and ma50 < ma200

            # MACD crossover
            macd_bull = macd_bear = False
            if all(v is not None for v in [macd_line, signal_line, macd_prev, signal_prev]):
                macd_bull = macd_prev <= signal_prev and macd_line > signal_line
                macd_bear = macd_prev >= signal_prev and macd_line < signal_line

            # Breakout (20-day high)
            high_20 = float(df["Close"].iloc[-21:-1].max()) if len(df) >= 21 else None
            low_20  = float(df["Close"].iloc[-21:-1].min()) if len(df) >= 21 else None
            breakout   = close_today > high_20 if high_20 else False
            breakdown  = close_today < low_20  if low_20  else False

            # BB Squeeze (width now < width 5 bars ago * 0.8)
            bb_squeeze = (bb_width < bb_width_prev * 0.8) if (bb_width and bb_width_prev) else False

            # Consecutive days up/down
            consec = 0
            direction = 1 if close.iloc[-1] > close.iloc[-2] else -1
            for i in range(len(close) - 1, 0, -1):
                if (close.iloc[i] - close.iloc[i-1]) * direction > 0:
                    consec += 1
                else:
                    break

            # Inside bar
            prev_high = float(df["High"].iloc[-2])
            prev_low  = float(df["Low"].iloc[-2])
            curr_high = float(df["High"].iloc[-1])
            curr_low  = float(df["Low"].iloc[-1])
            inside_bar = curr_high < prev_high and curr_low > prev_low

            signals.append({
                "ticker":        ticker.replace(".JK", ""),
                "rsi":           rsi,
                "ma20":          ma20,
                "ma50":          ma50,
                "ma200":         ma200,
                "close":         close_today,
                "golden_cross":  golden_cross,
                "death_cross":   death_cross,
                "macd_bull":     macd_bull,
                "macd_bear":     macd_bear,
                "breakout":      breakout,
                "breakdown":     breakdown,
                "bb_squeeze":    bb_squeeze,
                "bb_upper":      bb_upper,
                "bb_lower":      bb_lower,
                "atr":           atr_val,
                "consec_days":   consec * direction,
                "inside_bar":    inside_bar,
            })
        except Exception:
            pass
    return pd.DataFrame(signals)

# ─── WATCHLIST SCORING ────────────────────────────────────────────────────────
def compute_watchlist(movers_df, tech_df):
    """Score each ticker based on bullish signals for tomorrow."""
    if movers_df.empty or tech_df.empty:
        return pd.DataFrame(columns=["ticker", "score", "rsi", "change_pct", "vol_ratio", "breakout", "golden_cross"])
    merged = movers_df.merge(tech_df, on="ticker", how="inner")
    merged["score"] = 0

    # RSI oversold → +2
    merged.loc[merged["rsi"] < 30, "score"] += 2
    # RSI slightly oversold → +1
    merged.loc[(merged["rsi"] >= 30) & (merged["rsi"] < 40), "score"] += 1
    # Volume spike → +2
    merged.loc[merged["vol_ratio"] > 2, "score"] += 2
    # Breakout → +2
    merged.loc[merged["breakout"] == True, "score"] += 2
    # Golden cross → +2
    merged.loc[merged["golden_cross"] == True, "score"] += 2
    # MACD bullish crossover → +1
    merged.loc[merged["macd_bull"] == True, "score"] += 1
    # High close ratio (buyer control) → +1
    merged.loc[merged["hcr"] > 0.8, "score"] += 1
    # BB squeeze → +1 (potential breakout)
    merged.loc[merged["bb_squeeze"] == True, "score"] += 1
    # Price above MA20 → +1
    merged.loc[merged["close_x"] > merged["ma20"], "score"] += 1
    # Not overbought
    merged.loc[merged["rsi"] > 70, "score"] -= 2

    top = merged.nlargest(WATCHLIST_TOP_N, "score")[
        ["ticker", "score", "rsi", "change_pct", "vol_ratio", "breakout", "golden_cross"]
    ]
    return top

# ─── FORMAT MESSAGE ───────────────────────────────────────────────────────────
def fmt_pct(val, plus=True):
    if val is None: return "N/A"
    sign = "+" if val > 0 and plus else ""
    return f"{sign}{val:.2f}%"

def fmt_num(val, decimals=2):
    if val is None: return "N/A"
    return f"{val:,.{decimals}f}"

def build_message(ihsg, ihsg_pct, movers_df, tech_df, global_data, watchlist_df):
    today = datetime.now().strftime("%A, %d %B %Y")
    lines = []

    # Header
    ihsg_emoji = "🟢" if ihsg_pct and ihsg_pct > 0 else "🔴"
    lines.append(f"📊 *RINGKASAN PASAR IDX*")
    lines.append(f"📅 {today}\n")

    # 1. IHSG
    lines.append(f"*🇮🇩 IHSG*")
    if ihsg:
        lines.append(f"{fmt_num(ihsg)} {ihsg_emoji} {fmt_pct(ihsg_pct)}\n")
    else:
        lines.append("Data tidak tersedia\n")

    # 2. Advance / Decline
    if not movers_df.empty:
        adv = (movers_df["change_pct"] > 0).sum()
        dec = (movers_df["change_pct"] < 0).sum()
        flat = (movers_df["change_pct"] == 0).sum()
        lines.append(f"*📊 Advance / Decline*")
        lines.append(f"🟢 Naik: {adv}  🔴 Turun: {dec}  ⚪ Flat: {flat}\n")

    # 3. Top Gainers
    lines.append("*🚀 Top 5 Gainer*")
    top_gain = movers_df.nlargest(5, "change_pct")
    for _, r in top_gain.iterrows():
        lines.append(f"  `{r['ticker']}` +{r['change_pct']:.2f}%")
    lines.append("")

    # 4. Top Losers
    lines.append("*📉 Top 5 Loser*")
    top_lose = movers_df.nsmallest(5, "change_pct")
    for _, r in top_lose.iterrows():
        lines.append(f"  `{r['ticker']}` {r['change_pct']:.2f}%")
    lines.append("")

    # 5. Top Volume
    lines.append("*💹 Top 5 Volume*")
    top_vol = movers_df.nlargest(5, "volume")
    for _, r in top_vol.iterrows():
        lines.append(f"  `{r['ticker']}` {r['volume']/1e6:.1f}M lot")
    lines.append("")

    # 6. Unusual Activity
    lines.append("*🔍 Unusual Activity*")
    unusual = movers_df[movers_df["vol_ratio"] > 2].sort_values("vol_ratio", ascending=False).head(5)
    if unusual.empty:
        lines.append("  Tidak ada unusual volume hari ini")
    else:
        for _, r in unusual.iterrows():
            lines.append(f"  `{r['ticker']}` vol {r['vol_ratio']:.1f}x rata-rata | harga {fmt_pct(r['change_pct'])}")

    # ARA/ARB
    ara = movers_df[movers_df["change_pct"] >= 24.9]
    arb = movers_df[movers_df["change_pct"] <= -24.9]
    if not ara.empty:
        tickers_ara = ", ".join([f"`{t}`" for t in ara["ticker"]])
        lines.append(f"  🔺 ARA: {tickers_ara}")
    if not arb.empty:
        tickers_arb = ", ".join([f"`{t}`" for t in arb["ticker"]])
        lines.append(f"  🔻 ARB: {tickers_arb}")

    # Gap
    gap_up   = movers_df[movers_df["gap_pct"] > 2].sort_values("gap_pct", ascending=False).head(3)
    gap_down = movers_df[movers_df["gap_pct"] < -2].sort_values("gap_pct").head(3)
    if not gap_up.empty:
        g = ", ".join([f"`{r['ticker']}` +{r['gap_pct']:.1f}%" for _, r in gap_up.iterrows()])
        lines.append(f"  ⬆️ Gap Up: {g}")
    if not gap_down.empty:
        g = ", ".join([f"`{r['ticker']}` {r['gap_pct']:.1f}%" for _, r in gap_down.iterrows()])
        lines.append(f"  ⬇️ Gap Down: {g}")
    lines.append("")

    # 7. Sinyal Teknikal
    lines.append("*⚡ Sinyal Teknikal*")

    def safe_filter(df, col, condition_fn):
        if col not in df.columns or df.empty:
            return []
        try:
            return df[condition_fn(df[col])]["ticker"].tolist()
        except Exception:
            return []

    oversold   = safe_filter(tech_df, "rsi", lambda x: x < 30)
    overbought = safe_filter(tech_df, "rsi", lambda x: x > 70)
    golden     = safe_filter(tech_df, "golden_cross", lambda x: x == True)
    death      = safe_filter(tech_df, "death_cross", lambda x: x == True)
    macd_b     = safe_filter(tech_df, "macd_bull", lambda x: x == True)
    macd_br    = safe_filter(tech_df, "macd_bear", lambda x: x == True)
    breakouts  = safe_filter(tech_df, "breakout", lambda x: x == True)
    breakdowns = safe_filter(tech_df, "breakdown", lambda x: x == True)
    squeezes   = safe_filter(tech_df, "bb_squeeze", lambda x: x == True)

    def fmt_list(lst): return ", ".join([f"`{t}`" for t in lst]) if lst else "–"

    lines.append(f"  RSI Oversold (<30): {fmt_list(oversold)}")
    lines.append(f"  RSI Overbought (>70): {fmt_list(overbought)}")
    lines.append(f"  Golden Cross: {fmt_list(golden)}")
    lines.append(f"  Death Cross: {fmt_list(death)}")
    lines.append(f"  MACD Bullish: {fmt_list(macd_b)}")
    lines.append(f"  MACD Bearish: {fmt_list(macd_br)}")
    lines.append(f"  Breakout 20H: {fmt_list(breakouts)}")
    lines.append(f"  Breakdown 20L: {fmt_list(breakdowns)}")
    lines.append(f"  BB Squeeze: {fmt_list(squeezes)}")

    # Consecutive days
    if "consec_days" in tech_df.columns and not tech_df.empty:
        consec_up   = tech_df[tech_df["consec_days"] >= 4].sort_values("consec_days", ascending=False)
        consec_down = tech_df[tech_df["consec_days"] <= -4].sort_values("consec_days")
    else:
        consec_up = consec_down = tech_df.iloc[0:0]
    if not consec_up.empty:
        c = ", ".join([f"`{r['ticker']}` ({r['consec_days']}h)" for _, r in consec_up.iterrows()])
        lines.append(f"  📈 Naik {consec_up['consec_days'].min()}+ hari berturut: {c}")
    if not consec_down.empty:
        c = ", ".join([f"`{r['ticker']}` ({abs(r['consec_days'])}h)" for _, r in consec_down.iterrows()])
        lines.append(f"  📉 Turun {abs(consec_down['consec_days'].max())}+ hari berturut: {c}")
    lines.append("")

    # 8. 52-Week Proximity
    near_ath = movers_df[movers_df["pct_from_ath"] >= -2]
    near_atl = movers_df[movers_df["pct_from_atl"] <= 2]
    if not near_ath.empty or not near_atl.empty:
        lines.append("*🏆 52-Week Proximity*")
        if not near_ath.empty:
            t = ", ".join([f"`{r['ticker']}`" for _, r in near_ath.iterrows()])
            lines.append(f"  🔝 Dekat ATH: {t}")
        if not near_atl.empty:
            t = ", ".join([f"`{r['ticker']}`" for _, r in near_atl.iterrows()])
            lines.append(f"  🔻 Dekat ATL: {t}")
        lines.append("")

    # 9. Global
    lines.append("*🌍 Global & Komoditas*")
    def gfmt(name, data):
        if data is None: return f"  {name}: N/A"
        emoji = "🟢" if data["pct"] > 0 else "🔴"
        return f"  {name}: {fmt_num(data['value'])} {emoji} {fmt_pct(data['pct'])}"

    lines.append(gfmt("S&P500",    global_data.get("S&P500")))
    lines.append(gfmt("Nasdaq",    global_data.get("Nasdaq")))
    lines.append(gfmt("DJIA",      global_data.get("DJIA")))
    lines.append(gfmt("USD/IDR",   global_data.get("USD/IDR")))
    lines.append(gfmt("Emas",      global_data.get("Emas")))
    lines.append(gfmt("Crude Oil", global_data.get("Crude Oil")))
    lines.append(gfmt("Nikel",     global_data.get("Nikel")))
    lines.append(gfmt("US10Y",     global_data.get("US10Y")))
    lines.append("")

    # 10. Watchlist
    lines.append(f"*🎯 Watchlist Besok (Top {WATCHLIST_TOP_N})*")
    lines.append("_Berdasarkan scoring sinyal teknikal — bukan rekomendasi finansial_")
    for i, (_, r) in enumerate(watchlist_df.iterrows(), 1):
        tags = []
        if r.get("rsi") and r["rsi"] < 30: tags.append("RSI OS")
        if r.get("breakout"):               tags.append("Breakout")
        if r.get("golden_cross"):           tags.append("Golden X")
        if r.get("vol_ratio") and r["vol_ratio"] > 2: tags.append(f"Vol {r['vol_ratio']:.1f}x")
        tag_str = " | ".join(tags) if tags else "Multi-signal"
        lines.append(f"  {i}. `{r['ticker']}` — Skor {int(r['score'])} — {tag_str}")
    lines.append("")

    lines.append("_Data: yfinance | Dikirim otomatis jam 17.00 WIB_")
    lines.append("⚠️ _Bukan rekomendasi investasi. DYOR._")

    return "\n".join(lines)

# ─── SEND TELEGRAM ────────────────────────────────────────────────────────────
def send_telegram(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Split if too long (Telegram limit 4096 chars)
    max_len = 4000
    chunks = [message[i:i+max_len] for i in range(0, len(message), max_len)]
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id":    chat_id,
            "text":       chunk,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload)
        if not resp.ok:
            print(f"❌ Telegram error: {resp.text}")
        else:
            print(f"✅ Pesan {i+1}/{len(chunks)} terkirim")
        time.sleep(0.5)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    print(f"\n{'='*50}")
    print(f"🚀 IDX Daily Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    tickers = get_tickers()

    # Fetch
    raw = fetch_all_data(tickers)
    ticker_data = extract_ticker_data(raw, tickers)

    # Compute
    print("⚙️  Menghitung movers...")
    movers_df = compute_movers(ticker_data)

    print("⚙️  Menghitung sinyal teknikal...")
    tech_df = compute_technicals(ticker_data)

    print("⚙️  Membuat watchlist...")
    watchlist_df = compute_watchlist(movers_df, tech_df)

    print("🌍 Fetch data global...")
    ihsg, ihsg_pct = get_ihsg()
    global_data = get_global_data()

    # Build message
    print("📝 Menyusun pesan...")
    message = build_message(ihsg, ihsg_pct, movers_df, tech_df, global_data, watchlist_df)

    # Print to console (for testing)
    print("\n" + "="*50)
    print("PREVIEW PESAN:")
    print("="*50)
    print(message)

    # Send to Telegram
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(message, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    else:
        print("⚠️  TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset di .env")

if __name__ == "__main__":
    run()