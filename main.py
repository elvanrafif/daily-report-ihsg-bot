import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

from tickers import IDX_TICKERS

SEP = "━━━━━━━━━━━━━━━━━━━━━━"

def get_tickers():
    return [f"{t}.JK" for t in IDX_TICKERS]

# ─── FETCH ────────────────────────────────────────────────────────────────────
def fetch_all_data(tickers):
    print(f"📥 Fetching {len(tickers)} tickers (60d)...")
    raw = yf.download(
        tickers, period="60d", interval="1d",
        auto_adjust=True, progress=False, threads=True, multi_level_index=True,
    )
    return raw

def extract_ticker_data(raw, tickers):
    result = {}
    if raw is None or raw.empty:
        return result
    is_multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            if is_multi:
                if ticker in raw.columns.get_level_values(1):
                    df = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
            else:
                df = raw[ticker].copy() if ticker in raw.columns else None
                if df is None: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df.dropna(subset=["Close"])
            if len(df) >= 20:
                result[ticker] = df
        except Exception:
            pass
    print(f"✅ {len(result)} tickers berhasil diproses")
    return result

# ─── IHSG & GLOBAL ────────────────────────────────────────────────────────────
def get_ihsg():
    try:
        df = yf.download("^JKSE", period="5d", interval="1d",
                         progress=False, auto_adjust=True, multi_level_index=False)
        df = df.dropna()
        if len(df) >= 2:
            c, p = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
            return c, (c - p) / p * 100
    except Exception:
        pass
    return None, None

def fetch_single(sym):
    try:
        df = yf.download(sym, period="5d", interval="1d",
                         progress=False, auto_adjust=True, multi_level_index=False)
        df = df.dropna()
        if len(df) >= 2:
            c, p = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
            return {"value": c, "pct": (c - p) / p * 100}
    except Exception:
        pass
    return None

def get_global_data():
    symbols = {
        "S&P500": "^GSPC", "Nasdaq": "^IXIC", "DJIA": "^DJI",
        "USD/IDR": "IDR=X", "Emas": "GC=F", "Crude Oil": "CL=F", "US10Y": "^TNX",
    }
    return {name: fetch_single(sym) for name, sym in symbols.items()}

# ─── MOVERS ───────────────────────────────────────────────────────────────────
def compute_movers(ticker_data):
    rows = []
    for ticker, df in ticker_data.items():
        if len(df) < 2: continue
        try:
            ct   = float(df["Close"].iloc[-1])
            cp   = float(df["Close"].iloc[-2])
            vol  = float(df["Volume"].iloc[-1])
            vavg = float(df["Volume"].iloc[:-1].mean())
            h    = float(df["High"].iloc[-1])
            l    = float(df["Low"].iloc[-1])
            o    = float(df["Open"].iloc[-1])
            rng  = h - l

            # Nilai transaksi hari ini (harga × volume)
            nilai = ct * vol

            # Performance 1 bulan (~21 trading days)
            perf_1m = None
            if len(df) >= 22:
                p1m    = float(df["Close"].iloc[-22])
                perf_1m = (ct - p1m) / p1m * 100

            # ADR 14 hari
            adr = None
            if len(df) >= 14:
                dr  = (df["High"].tail(14) - df["Low"].tail(14)) / df["Close"].tail(14) * 100
                adr = float(dr.mean())

            rows.append({
                "ticker":     ticker.replace(".JK", ""),
                "close":      ct,
                "change_pct": (ct - cp) / cp * 100,
                "volume":     vol,
                "nilai":      nilai,
                "vol_ratio":  vol / vavg if vavg > 0 else 0,
                "gap_pct":    (o - cp) / cp * 100,
                "hcr":        (ct - l) / rng if rng > 0 else 0.5,
                "perf_1m":    perf_1m,
                "adr":        adr,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

# ─── TECHNICALS ───────────────────────────────────────────────────────────────
def compute_technicals(ticker_data):
    signals = []
    for ticker, df in ticker_data.items():
        if len(df) < 26: continue
        try:
            close = df["Close"].squeeze().astype(float)
            high  = df["High"].squeeze().astype(float)
            low   = df["Low"].squeeze().astype(float)
            if not isinstance(close, pd.Series) or len(close) < 26: continue

            # RSI
            rsi_s = ta.rsi(close, length=14)
            rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.isna().all() else None

            # MA
            ma20 = float(ta.sma(close, length=20).iloc[-1])
            ma50 = float(ta.sma(close, length=50).iloc[-1]) if len(df) >= 50 else None
            ma200= float(ta.sma(close, length=200).iloc[-1]) if len(df) >= 200 else None

            # SMA alignment (dari screener)
            cn = float(close.iloc[-1])
            sma_full    = False
            sma_relaxed = False
            if ma50 and ma200:
                sma_relaxed = cn > ma50 > ma200
                if ma20:
                    sma_full = cn > ma20 > ma50 > ma200

            # MACD
            macd_df   = ta.macd(close)
            macd_bull = macd_bear = False
            if macd_df is not None and len(macd_df) >= 2:
                cols = macd_df.columns.tolist()
                mc = [c for c in cols if c.startswith("MACD_") and "h" not in c.lower() and "s" not in c.lower()]
                sc = [c for c in cols if "MACDs_" in c]
                if mc and sc:
                    ml, sl   = float(macd_df[mc[0]].iloc[-1]), float(macd_df[sc[0]].iloc[-1])
                    ml2, sl2 = float(macd_df[mc[0]].iloc[-2]), float(macd_df[sc[0]].iloc[-2])
                    macd_bull = ml2 <= sl2 and ml > sl
                    macd_bear = ml2 >= sl2 and ml < sl

            # BB Squeeze
            bb = ta.bbands(close, length=20)
            bb_squeeze = False
            if bb is not None and len(bb) >= 6:
                bwc = [c for c in bb.columns if "BBB_" in c]
                if bwc:
                    bb_squeeze = float(bb[bwc[0]].iloc[-1]) < float(bb[bwc[0]].iloc[-6]) * 0.8

            # Breakout/Breakdown 20H
            breakout = breakdown = False
            if len(df) >= 21:
                breakout  = cn > float(close.iloc[-21:-1].max())
                breakdown = cn < float(close.iloc[-21:-1].min())

            # Golden/Death cross MA20 vs MA50
            golden_cross = death_cross = False
            if ma50 is not None and len(df) >= 51:
                ma20p = float(ta.sma(close, length=20).iloc[-2])
                ma50p = float(ta.sma(close, length=50).iloc[-2])
                golden_cross = ma20p <= ma50p and ma20 > ma50
                death_cross  = ma20p >= ma50p and ma20 < ma50

            # Consecutive days
            direction = 1 if float(close.iloc[-1]) > float(close.iloc[-2]) else -1
            consec = 0
            for i in range(len(close) - 1, 0, -1):
                if (float(close.iloc[i]) - float(close.iloc[i-1])) * direction > 0:
                    consec += 1
                else:
                    break

            signals.append({
                "ticker": ticker.replace(".JK", ""),
                "rsi": rsi, "ma20": ma20, "ma50": ma50, "ma200": ma200, "close": cn,
                "sma_full": sma_full, "sma_relaxed": sma_relaxed,
                "golden_cross": golden_cross, "death_cross": death_cross,
                "macd_bull": macd_bull, "macd_bear": macd_bear,
                "breakout": breakout, "breakdown": breakdown,
                "bb_squeeze": bb_squeeze, "consec_days": consec * direction,
            })
        except Exception:
            pass

    cols = ["ticker","rsi","ma20","ma50","ma200","close","sma_full","sma_relaxed",
            "golden_cross","death_cross","macd_bull","macd_bear",
            "breakout","breakdown","bb_squeeze","consec_days"]
    return pd.DataFrame(signals) if signals else pd.DataFrame(columns=cols)

# ─── WATCHLIST (TIER A+/A/B dari screener) ────────────────────────────────────
def compute_watchlist(movers_df, tech_df):
    if movers_df.empty or tech_df.empty:
        return {"A+": [], "A": [], "B": []}

    merged = movers_df.merge(tech_df, on="ticker", how="inner")
    if merged.empty:
        return {"A+": [], "A": [], "B": []}

    tiers = {"A+": [], "A": [], "B": []}

    for _, r in merged.iterrows():
        perf_1m  = r.get("perf_1m")
        adr      = r.get("adr") or 0
        nilai    = r.get("nilai") or 0
        sma_full = r.get("sma_full", False)
        sma_rel  = r.get("sma_relaxed", False)
        ticker   = r["ticker"]

        if perf_1m is None:
            continue

        # Tier A+: momentum super kuat
        if (perf_1m > 30 and nilai > 1e9 and adr > 4.0 and sma_full):
            tiers["A+"].append(ticker)
        # Tier A: momentum kuat
        elif (perf_1m > 15 and nilai > 500e6 and adr > 3.0 and sma_full):
            tiers["A"].append(ticker)
        # Tier B: early detection
        elif (perf_1m > 5 and nilai > 250e6 and adr > 2.5 and sma_rel):
            tiers["B"].append(ticker)

    # Sort alphabetically
    for t in tiers:
        tiers[t].sort()

    return tiers

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def fmt_pct(val):
    if val is None: return "N/A"
    return f"+{val:.2f}%" if val > 0 else f"{val:.2f}%"

def fmt_num(val, dec=2):
    if val is None: return "N/A"
    return f"{val:,.{dec}f}"

def fmt_nilai(val):
    """Format nilai transaksi ke Rp T / M / K."""
    if val is None: return "N/A"
    if val >= 1e12: return f"Rp {val/1e12:.2f}T"
    if val >= 1e9:  return f"Rp {val/1e9:.1f}M"
    if val >= 1e6:  return f"Rp {val/1e6:.0f}rb"
    return f"Rp {val:,.0f}"

def safe_filter(df, col, fn):
    if df.empty or col not in df.columns: return []
    try: return df[fn(df[col])]["ticker"].tolist()
    except: return []

def fmt_ticker_rows(lst, per_row=5):
    """Format list ticker jadi baris-baris, per_row ticker per baris."""
    if not lst: return ["–"]
    rows = []
    for i in range(0, len(lst), per_row):
        chunk = lst[i:i+per_row]
        rows.append(" ".join([f"`{t}`" for t in chunk]))
    return rows

# ─── BUILD MESSAGE ────────────────────────────────────────────────────────────
def build_message(ihsg, ihsg_pct, movers_df, tech_df, global_data, watchlist_tiers):
    today = datetime.now().strftime("%A, %d %B %Y")
    L = []

    # Header
    L.append("📊 *RINGKASAN PASAR IDX*")
    L.append(f"📅 {today}")
    L.append("")

    # IHSG + A/D
    if ihsg:
        em = "🟢" if ihsg_pct > 0 else "🔴"
        L.append(f"*🇮🇩 IHSG:* {fmt_num(ihsg, 0)} {em} {fmt_pct(ihsg_pct)}")
    else:
        L.append("*🇮🇩 IHSG:* Tidak tersedia")

    if not movers_df.empty:
        adv  = int((movers_df["change_pct"] > 0).sum())
        dec  = int((movers_df["change_pct"] < 0).sum())
        flat = int((movers_df["change_pct"] == 0).sum())
        L.append(f"*📊 A/D:* 🟢 {adv} emiten naik  🔴 {dec} emiten turun  ⚪ {flat} emiten flat")
    L.append(SEP)

    # Top 5 Gainer & Loser
    L.append("*🚀 Top 5 Gainer*")
    if not movers_df.empty:
        for _, r in movers_df.nlargest(5, "change_pct").iterrows():
            L.append(f"  `{r['ticker']}` {fmt_pct(r['change_pct'])}")
    L.append("")

    L.append("*📉 Top 5 Loser*")
    if not movers_df.empty:
        for _, r in movers_df.nsmallest(5, "change_pct").iterrows():
            L.append(f"  `{r['ticker']}` {fmt_pct(r['change_pct'])}")
    L.append(SEP)

    # Top Nilai Transaksi
    L.append("*💹 Top 5 Nilai Transaksi Harian*")
    if not movers_df.empty and "nilai" in movers_df.columns:
        for _, r in movers_df.nlargest(5, "nilai").iterrows():
            L.append(f"  `{r['ticker']}` {fmt_nilai(r['nilai'])}")
    L.append(SEP)

    # Unusual Activity
    L.append("*🔍 Unusual Activity*")
    if not movers_df.empty:
        unusual = movers_df[movers_df["vol_ratio"] > 2].sort_values("vol_ratio", ascending=False).head(5)
        if unusual.empty:
            L.append("  Tidak ada unusual volume hari ini")
        else:
            for _, r in unusual.iterrows():
                L.append(f"  `{r['ticker']}` {r['vol_ratio']:.1f}x rata-rata | {fmt_pct(r['change_pct'])}")

        ara = movers_df[movers_df["change_pct"] >= 24.9]
        arb = movers_df[movers_df["change_pct"] <= -24.9]
        if not ara.empty:
            L.append(f"  🔺 ARA: {' '.join([f'`{t}`' for t in ara['ticker']])}")
        if not arb.empty:
            L.append(f"  🔻 ARB: {' '.join([f'`{t}`' for t in arb['ticker']])}")

        gu = movers_df[movers_df["gap_pct"] > 2].sort_values("gap_pct", ascending=False).head(3)
        gd = movers_df[movers_df["gap_pct"] < -2].sort_values("gap_pct").head(3)
        if not gu.empty:
            L.append("  ⬆️ Gap Up: " + "  ".join([f"`{r['ticker']}` +{r['gap_pct']:.1f}%" for _, r in gu.iterrows()]))
        if not gd.empty:
            L.append("  ⬇️ Gap Down: " + "  ".join([f"`{r['ticker']}` {r['gap_pct']:.1f}%" for _, r in gd.iterrows()]))
    L.append(SEP)

    # Top Momentum 1 Bulan
    L.append("*📈 Top 5 Momentum 1 Bulan*")
    if not movers_df.empty and "perf_1m" in movers_df.columns:
        top_mom = movers_df.dropna(subset=["perf_1m"]).nlargest(5, "perf_1m")
        for _, r in top_mom.iterrows():
            L.append(f"  `{r['ticker']}` {fmt_pct(r['perf_1m'])}")
    L.append(SEP)

    # Sinyal Teknikal
    L.append("*⚡ Sinyal Teknikal*")
    if tech_df.empty:
        L.append("  Data tidak tersedia")
    else:
        def add_signal(label, lst):
            if lst:
                L.append(f"\n{label}")
                for row in fmt_ticker_rows(lst):
                    L.append(f"  {row}")
            else:
                L.append(f"{label} –")

        oversold   = safe_filter(tech_df, "rsi",          lambda x: x < 30)
        overbought = safe_filter(tech_df, "rsi",          lambda x: x > 70)
        golden     = safe_filter(tech_df, "golden_cross", lambda x: x == True)
        death      = safe_filter(tech_df, "death_cross",  lambda x: x == True)
        macd_b     = safe_filter(tech_df, "macd_bull",    lambda x: x == True)
        macd_br    = safe_filter(tech_df, "macd_bear",    lambda x: x == True)
        breakouts  = safe_filter(tech_df, "breakout",     lambda x: x == True)
        breakdowns = safe_filter(tech_df, "breakdown",    lambda x: x == True)
        squeezes   = safe_filter(tech_df, "bb_squeeze",   lambda x: x == True)

        add_signal("RSI Oversold (<30):", oversold)
        add_signal("RSI Overbought (>70):", overbought)
        add_signal("Golden Cross (MA20>MA50):", golden)
        add_signal("Death Cross:", death)
        add_signal("MACD Bullish:", macd_b)
        add_signal("MACD Bearish:", macd_br)
        add_signal("Breakout 20H:", breakouts)
        add_signal("Breakdown 20L:", breakdowns)
        add_signal("BB Squeeze:", squeezes)

        if "consec_days" in tech_df.columns:
            cup   = tech_df[tech_df["consec_days"] >= 4].sort_values("consec_days", ascending=False)
            cdown = tech_df[tech_df["consec_days"] <= -4].sort_values("consec_days")
            if not cup.empty:
                L.append("\n📈 Naik berturut-turut:")
                items = [f"`{r['ticker']}` ({r['consec_days']}h)" for _, r in cup.iterrows()]
                for row in fmt_ticker_rows(items, per_row=4):
                    L.append(f"  {row}")
            if not cdown.empty:
                L.append("\n📉 Turun berturut-turut:")
                items = [f"`{r['ticker']}` ({abs(r['consec_days'])}h)" for _, r in cdown.iterrows()]
                for row in fmt_ticker_rows(items, per_row=4):
                    L.append(f"  {row}")
    L.append(SEP)

    # Global & Komoditas
    L.append("*🌍 Global & Komoditas*")
    def gfmt(name, data):
        if not data: return f"  {name}: N/A"
        em = "🟢" if data["pct"] > 0 else "🔴"
        return f"  {name}: {fmt_num(data['value'])} {em} {fmt_pct(data['pct'])}"
    for name in ["S&P500","Nasdaq","DJIA","USD/IDR","Emas","Crude Oil","US10Y"]:
        L.append(gfmt(name, global_data.get(name)))
    L.append(SEP)

    # Watchlist Tier A+/A/B
    L.append("*🎯 Watchlist Besok*")
    L.append("_Screening otomatis — bukan rekomendasi finansial_")
    L.append("")

    ap = watchlist_tiers.get("A+", [])
    a  = watchlist_tiers.get("A",  [])
    b  = watchlist_tiers.get("B",  [])

    if ap:
        L.append("⭐ *A+ (Super Momentum):*")
        for row in fmt_ticker_rows(ap):
            L.append(f"  {row}")
    else:
        L.append("⭐ *A+:* –")

    if a:
        L.append("✅ *A (Momentum Kuat):*")
        for row in fmt_ticker_rows(a):
            L.append(f"  {row}")
    else:
        L.append("✅ *A:* –")

    if b:
        L.append("🔵 *B (Early Detection):*")
        for row in fmt_ticker_rows(b):
            L.append(f"  {row}")
    else:
        L.append("🔵 *B:* –")

    L.append(SEP)
    L.append("_Data: yfinance | Jam 17.00 WIB_")
    L.append("⚠️ _Bukan rekomendasi investasi. DYOR._")

    return "\n".join(L)

# ─── SEND TELEGRAM ────────────────────────────────────────────────────────────
def send_telegram(message, token, chat_id):
    url    = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for i, chunk in enumerate(chunks):
        resp = requests.post(url, json={
            "chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"
        })
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

    tickers     = get_tickers()
    raw         = fetch_all_data(tickers)
    ticker_data = extract_ticker_data(raw, tickers)

    print("⚙️  Menghitung movers...")
    movers_df = compute_movers(ticker_data)

    print("⚙️  Menghitung sinyal teknikal...")
    tech_df = compute_technicals(ticker_data)
    print(f"   → {len(tech_df)} ticker berhasil dihitung sinyalnya")

    print("⚙️  Membuat watchlist tier...")
    watchlist_tiers = compute_watchlist(movers_df, tech_df)
    for tier, lst in watchlist_tiers.items():
        print(f"   {tier}: {len(lst)} saham")

    print("🌍 Fetch data global...")
    ihsg, ihsg_pct = get_ihsg()
    global_data    = get_global_data()

    print("📝 Menyusun pesan...")
    message = build_message(ihsg, ihsg_pct, movers_df, tech_df, global_data, watchlist_tiers)

    print("\n" + "="*50)
    print("PREVIEW PESAN:")
    print("="*50)
    print(message)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(message, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    else:
        print("⚠️  TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset")

if __name__ == "__main__":
    run()