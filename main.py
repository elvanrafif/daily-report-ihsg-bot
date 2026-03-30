import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WATCHLIST_TOP_N  = int(os.getenv("WATCHLIST_TOP_N", 5))

from tickers import IDX_TICKERS

def get_tickers():
    return [f"{t}.JK" for t in IDX_TICKERS]

def fetch_all_data(tickers):
    print(f"📥 Fetching {len(tickers)} tickers...")
    raw = yf.download(
        tickers, period="30d", interval="1d",
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
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker].copy()
                else:
                    continue
            else:
                if ticker in raw.columns:
                    df = raw[ticker].copy()
                else:
                    continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df.dropna(subset=["Close"])
            if len(df) >= 15:
                result[ticker] = df
        except Exception:
            pass
    print(f"✅ {len(result)} tickers berhasil diproses")
    return result

def get_ihsg():
    for sym in ["^JKSE"]:
        try:
            df = yf.download(sym, period="5d", interval="1d",
                             progress=False, auto_adjust=True, multi_level_index=False)
            df = df.dropna()
            if len(df) >= 2:
                c = float(df["Close"].iloc[-1])
                p = float(df["Close"].iloc[-2])
                return c, (c - p) / p * 100
        except Exception:
            continue
    return None, None

def fetch_single(sym):
    try:
        df = yf.download(sym, period="5d", interval="1d",
                         progress=False, auto_adjust=True, multi_level_index=False)
        df = df.dropna()
        if len(df) >= 2:
            c = float(df["Close"].iloc[-1])
            p = float(df["Close"].iloc[-2])
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

def compute_movers(ticker_data):
    rows = []
    for ticker, df in ticker_data.items():
        if len(df) < 2:
            continue
        try:
            ct   = float(df["Close"].iloc[-1])
            cp   = float(df["Close"].iloc[-2])
            vol  = float(df["Volume"].iloc[-1])
            vavg = float(df["Volume"].iloc[:-1].mean())
            h    = float(df["High"].iloc[-1])
            l    = float(df["Low"].iloc[-1])
            o    = float(df["Open"].iloc[-1])
            rng  = h - l
            rows.append({
                "ticker":     ticker.replace(".JK", ""),
                "close":      ct,
                "change_pct": (ct - cp) / cp * 100,
                "volume":     vol,
                "vol_ratio":  vol / vavg if vavg > 0 else 0,
                "gap_pct":    (o - cp) / cp * 100,
                "hcr":        (ct - l) / rng if rng > 0 else 0.5,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

def compute_technicals(ticker_data):
    signals = []
    for ticker, df in ticker_data.items():
        if len(df) < 20:
            continue
        try:
            close = df["Close"].squeeze().astype(float)
            high  = df["High"].squeeze().astype(float)
            low   = df["Low"].squeeze().astype(float)

            if not isinstance(close, pd.Series) or len(close) < 20:
                continue

            # RSI
            rsi_s = ta.rsi(close, length=14)
            rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.isna().all() else None

            # MA
            ma20 = float(ta.sma(close, length=20).iloc[-1])
            ma50 = float(ta.sma(close, length=50).iloc[-1]) if len(df) >= 50 else None

            # MACD
            macd_df   = ta.macd(close)
            macd_bull = macd_bear = False
            if macd_df is not None and len(macd_df) >= 2:
                cols  = macd_df.columns.tolist()
                mc    = [c for c in cols if c.startswith("MACD_") and "h" not in c.lower() and "s" not in c.lower()]
                sc    = [c for c in cols if "MACDs_" in c]
                if mc and sc:
                    ml, sl   = float(macd_df[mc[0]].iloc[-1]), float(macd_df[sc[0]].iloc[-1])
                    ml2, sl2 = float(macd_df[mc[0]].iloc[-2]), float(macd_df[sc[0]].iloc[-2])
                    macd_bull = ml2 <= sl2 and ml > sl
                    macd_bear = ml2 >= sl2 and ml < sl

            # BB Squeeze
            bb         = ta.bbands(close, length=20)
            bb_squeeze = False
            if bb is not None and len(bb) >= 6:
                bwc = [c for c in bb.columns if "BBB_" in c]
                if bwc:
                    bb_squeeze = float(bb[bwc[0]].iloc[-1]) < float(bb[bwc[0]].iloc[-6]) * 0.8

            # Breakout/Breakdown
            cn       = float(close.iloc[-1])
            breakout = breakdown = False
            if len(df) >= 21:
                breakout  = cn > float(close.iloc[-21:-1].max())
                breakdown = cn < float(close.iloc[-21:-1].min())

            # Golden/Death cross (MA20 vs MA50)
            golden_cross = death_cross = False
            if ma50 is not None and len(df) >= 51:
                ma20p = float(ta.sma(close, length=20).iloc[-2])
                ma50p = float(ta.sma(close, length=50).iloc[-2])
                golden_cross = ma20p <= ma50p and ma20 > ma50
                death_cross  = ma20p >= ma50p and ma20 < ma50

            # Consecutive days
            direction = 1 if float(close.iloc[-1]) > float(close.iloc[-2]) else -1
            consec    = 0
            for i in range(len(close) - 1, 0, -1):
                if (float(close.iloc[i]) - float(close.iloc[i-1])) * direction > 0:
                    consec += 1
                else:
                    break

            signals.append({
                "ticker": ticker.replace(".JK", ""), "rsi": rsi,
                "ma20": ma20, "ma50": ma50, "close": cn,
                "golden_cross": golden_cross, "death_cross": death_cross,
                "macd_bull": macd_bull, "macd_bear": macd_bear,
                "breakout": breakout, "breakdown": breakdown,
                "bb_squeeze": bb_squeeze, "consec_days": consec * direction,
            })
        except Exception:
            pass

    cols = ["ticker","rsi","ma20","ma50","close","golden_cross","death_cross",
            "macd_bull","macd_bear","breakout","breakdown","bb_squeeze","consec_days"]
    return pd.DataFrame(signals) if signals else pd.DataFrame(columns=cols)

def compute_watchlist(movers_df, tech_df):
    if movers_df.empty or tech_df.empty:
        return pd.DataFrame()
    merged = movers_df.merge(tech_df, on="ticker", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["score"] = 0
    merged.loc[merged["rsi"] < 30,                "score"] += 2
    merged.loc[(merged["rsi"] >= 30) & (merged["rsi"] < 40), "score"] += 1
    merged.loc[merged["vol_ratio"] > 2,           "score"] += 2
    merged.loc[merged["breakout"] == True,         "score"] += 2
    merged.loc[merged["golden_cross"] == True,     "score"] += 2
    merged.loc[merged["macd_bull"] == True,        "score"] += 1
    merged.loc[merged["hcr"] > 0.8,               "score"] += 1
    merged.loc[merged["bb_squeeze"] == True,       "score"] += 1
    merged.loc[merged["close_x"] > merged["ma20"], "score"] += 1
    merged.loc[merged["rsi"] > 70,                "score"] -= 2
    cols = ["ticker","score","rsi","change_pct","vol_ratio","breakout","golden_cross","macd_bull"]
    return merged.nlargest(WATCHLIST_TOP_N, "score")[cols]

def fmt_pct(val):
    if val is None: return "N/A"
    return f"+{val:.2f}%" if val > 0 else f"{val:.2f}%"

def fmt_num(val, dec=2):
    if val is None: return "N/A"
    return f"{val:,.{dec}f}"

def safe_filter(df, col, fn):
    if df.empty or col not in df.columns: return []
    try: return df[fn(df[col])]["ticker"].tolist()
    except: return []

def fmt_list(lst):
    if not lst: return "–"
    return ", ".join([f"`{t}`" for t in lst[:20]])

def build_message(ihsg, ihsg_pct, movers_df, tech_df, global_data, watchlist_df):
    today = datetime.now().strftime("%A, %d %B %Y")
    L = []

    L.append("📊 *RINGKASAN PASAR IDX*")
    L.append(f"📅 {today}")
    L.append("")

    # IHSG
    if ihsg:
        em = "🟢" if ihsg_pct > 0 else "🔴"
        L.append(f"*🇮🇩 IHSG:* {fmt_num(ihsg, 0)} {em} {fmt_pct(ihsg_pct)}")
    else:
        L.append("*🇮🇩 IHSG:* Tidak tersedia")

    # A/D
    if not movers_df.empty:
        adv  = (movers_df["change_pct"] > 0).sum()
        dec  = (movers_df["change_pct"] < 0).sum()
        flat = (movers_df["change_pct"] == 0).sum()
        L.append(f"*📊 A/D:* 🟢{adv} naik  🔴{dec} turun  ⚪{flat} flat")
    L.append("")

    # Movers
    L.append("*🚀 Top 5 Gainer*")
    for _, r in movers_df.nlargest(5, "change_pct").iterrows():
        L.append(f"  `{r['ticker']}` {fmt_pct(r['change_pct'])}")
    L.append("")

    L.append("*📉 Top 5 Loser*")
    for _, r in movers_df.nsmallest(5, "change_pct").iterrows():
        L.append(f"  `{r['ticker']}` {fmt_pct(r['change_pct'])}")
    L.append("")

    # Volume harian
    L.append("*💹 Top 5 Volume Harian*")
    for _, r in movers_df.nlargest(5, "volume").iterrows():
        L.append(f"  `{r['ticker']}` {r['volume']/1e6:.1f}M lot")
    L.append("")

    # Unusual Activity
    L.append("*🔍 Unusual Activity*")
    unusual = movers_df[movers_df["vol_ratio"] > 2].sort_values("vol_ratio", ascending=False).head(5)
    if unusual.empty:
        L.append("  Tidak ada unusual volume hari ini")
    else:
        for _, r in unusual.iterrows():
            L.append(f"  `{r['ticker']}` {r['vol_ratio']:.1f}x rata-rata | {fmt_pct(r['change_pct'])}")

    ara = movers_df[movers_df["change_pct"] >= 24.9]
    arb = movers_df[movers_df["change_pct"] <= -24.9]
    if not ara.empty:
        L.append(f"  🔺 ARA: {', '.join([f'`{t}`' for t in ara['ticker']])}")
    if not arb.empty:
        L.append(f"  🔻 ARB: {', '.join([f'`{t}`' for t in arb['ticker']])}")

    gu = movers_df[movers_df["gap_pct"] > 2].sort_values("gap_pct", ascending=False).head(3)
    gd = movers_df[movers_df["gap_pct"] < -2].sort_values("gap_pct").head(3)
    if not gu.empty:
        L.append("  ⬆️ Gap Up: " + ", ".join([f"`{r['ticker']}` +{r['gap_pct']:.1f}%" for _, r in gu.iterrows()]))
    if not gd.empty:
        L.append("  ⬇️ Gap Down: " + ", ".join([f"`{r['ticker']}` {r['gap_pct']:.1f}%" for _, r in gd.iterrows()]))
    L.append("")

    # Sinyal Teknikal
    L.append("*⚡ Sinyal Teknikal*")
    if tech_df.empty:
        L.append("  Data tidak tersedia")
    else:
        oversold   = safe_filter(tech_df, "rsi", lambda x: x < 30)
        overbought = safe_filter(tech_df, "rsi", lambda x: x > 70)
        golden     = safe_filter(tech_df, "golden_cross", lambda x: x == True)
        death      = safe_filter(tech_df, "death_cross",  lambda x: x == True)
        macd_b     = safe_filter(tech_df, "macd_bull",    lambda x: x == True)
        macd_br    = safe_filter(tech_df, "macd_bear",    lambda x: x == True)
        breakouts  = safe_filter(tech_df, "breakout",     lambda x: x == True)
        breakdowns = safe_filter(tech_df, "breakdown",    lambda x: x == True)
        squeezes   = safe_filter(tech_df, "bb_squeeze",   lambda x: x == True)
        L.append(f"  RSI Oversold (<30):    {fmt_list(oversold)}")
        L.append(f"  RSI Overbought (>70):  {fmt_list(overbought)}")
        L.append(f"  Golden Cross:          {fmt_list(golden)}")
        L.append(f"  Death Cross:           {fmt_list(death)}")
        L.append(f"  MACD Bullish:          {fmt_list(macd_b)}")
        L.append(f"  MACD Bearish:          {fmt_list(macd_br)}")
        L.append(f"  Breakout 20H:          {fmt_list(breakouts)}")
        L.append(f"  Breakdown 20L:         {fmt_list(breakdowns)}")
        L.append(f"  BB Squeeze:            {fmt_list(squeezes)}")
        if "consec_days" in tech_df.columns:
            cup   = tech_df[tech_df["consec_days"] >= 4].sort_values("consec_days", ascending=False)
            cdown = tech_df[tech_df["consec_days"] <= -4].sort_values("consec_days")
            if not cup.empty:
                L.append("  📈 Naik berturut: " + ", ".join([f"`{r['ticker']}` ({r['consec_days']}h)" for _, r in cup.iterrows()]))
            if not cdown.empty:
                L.append("  📉 Turun berturut: " + ", ".join([f"`{r['ticker']}` ({abs(r['consec_days'])}h)" for _, r in cdown.iterrows()]))
    L.append("")

    # Global
    L.append("*🌍 Global & Komoditas*")
    def gfmt(name, data):
        if not data: return f"  {name}: N/A"
        em = "🟢" if data["pct"] > 0 else "🔴"
        return f"  {name}: {fmt_num(data['value'])} {em} {fmt_pct(data['pct'])}"
    for name in ["S&P500","Nasdaq","DJIA","USD/IDR","Emas","Crude Oil","US10Y"]:
        L.append(gfmt(name, global_data.get(name)))
    L.append("")

    # Watchlist
    L.append(f"*🎯 Watchlist Besok (Top {WATCHLIST_TOP_N})*")
    L.append("_Scoring teknikal — bukan rekomendasi finansial_")
    if watchlist_df.empty:
        L.append("  Data tidak tersedia")
    else:
        for i, (_, r) in enumerate(watchlist_df.iterrows(), 1):
            tags = []
            rv = r.get("rsi")
            if rv is not None and not pd.isna(rv) and rv < 30: tags.append("RSI OS")
            if r.get("breakout"):     tags.append("Breakout")
            if r.get("golden_cross"): tags.append("Golden X")
            if r.get("macd_bull"):    tags.append("MACD Bull")
            vr = r.get("vol_ratio")
            if vr and vr > 2:         tags.append(f"Vol {vr:.1f}x")
            L.append(f"  {i}. `{r['ticker']}` — Skor {int(r['score'])} — {' | '.join(tags) if tags else 'Multi-signal'}")
    L.append("")
    L.append("_Data: yfinance | Jam 17.00 WIB_")
    L.append("⚠️ _Bukan rekomendasi investasi. DYOR._")

    return "\n".join(L)

def send_telegram(message, token, chat_id):
    url    = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for i, chunk in enumerate(chunks):
        resp = requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
        if not resp.ok:
            print(f"❌ Telegram error: {resp.text}")
        else:
            print(f"✅ Pesan {i+1}/{len(chunks)} terkirim")
        time.sleep(0.5)

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

    print("⚙️  Membuat watchlist...")
    watchlist_df = compute_watchlist(movers_df, tech_df)

    print("🌍 Fetch data global...")
    ihsg, ihsg_pct = get_ihsg()
    global_data    = get_global_data()

    print("📝 Menyusun pesan...")
    message = build_message(ihsg, ihsg_pct, movers_df, tech_df, global_data, watchlist_df)

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