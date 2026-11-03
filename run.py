import time
from datetime import datetime, timezone

from main import scan_market
from notifier import send_signal

MIN_SIGNAL_SCORE = 65
SCAN_INTERVAL = 300


def run():
    print("TAFI AI ROBOT STARTED")

    while True:
        try:
            print(
                "\nScanning:",
                datetime.now(timezone.utc)
            )

            signals = scan_market()

            sent_count = 0

            for signal in signals:
                if signal["score"] >= MIN_SIGNAL_SCORE:
                    if send_signal(signal):
                        sent_count += 1

            print("Signals found:", len(signals))
            print("Notifications sent:", sent_count)

        except Exception as error:
            print("TAFI ERROR:", error)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
