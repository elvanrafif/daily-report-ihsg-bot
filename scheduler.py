import schedule
import time
from datetime import datetime
from main import run

def job():
    print(f"⏰ Trigger jam {datetime.now().strftime('%H:%M:%S')} WIB")
    run()

# Jam 17.00 WIB = 10.00 UTC
schedule.every().day.at("10:00").do(job)

print("🤖 IDX Bot scheduler aktif — menunggu jam 17.00 WIB...")
print(f"   Sekarang: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run immediately on start for testing (comment out in production)
# run()

while True:
    schedule.run_pending()
    time.sleep(30)
