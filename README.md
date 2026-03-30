# IDX Daily Report Bot 📊

Bot Telegram yang mengirim ringkasan pasar IDX setiap hari jam 17.00 WIB.

## Fitur
- IHSG overview + advance/decline ratio
- Top 5 gainer, loser, volume
- Unusual activity (volume spike, ARA/ARB, gap up/down)
- Sinyal teknikal (RSI, MA cross, MACD, Bollinger Band, Breakout)
- 52-week high/low proximity
- Data global (S&P500, Nasdaq, USD/IDR, komoditas)
- Watchlist Top 5 besok (scoring multi-signal)

## Setup Lokal

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env dengan token Telegram kamu
python main.py        # test sekali jalan
python scheduler.py   # jalankan scheduler
```

## Setup Telegram Bot

1. Chat [@BotFather](https://t.me/BotFather) di Telegram
2. Ketik `/newbot` → ikuti instruksi → dapat TOKEN
3. Tambahkan bot ke grup kamu sebagai admin
4. Dapatkan CHAT_ID grup: forward pesan dari grup ke [@userinfobot](https://t.me/userinfobot)
5. Isi `.env` dengan TOKEN dan CHAT_ID

## Deploy ke Coolify

1. Push repo ini ke GitHub (private repo aman)
2. Di Coolify → New Resource → Docker
3. Connect ke GitHub repo ini
4. Set environment variables:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Deploy → selesai!

## Ganti Universe Saham

Edit `LQ45_TICKERS` di `main.py` dengan list ticker yang kamu mau.
Untuk semua saham IDX, bisa fetch dari IDX.co.id atau pakai CSV dari sini:
https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham

## Catatan

- Data dari yfinance, delay ~15 menit dari harga real-time
- Foreign flow & market turnover tidak tersedia di yfinance
- Bukan rekomendasi investasi — DYOR!
