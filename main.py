import time
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

BASE = "https://fapi.binance.com"
WORKERS = 8
TOP_RESULTS = 20
SCAN_INTERVAL = 300

session = requests.Session()
session.headers.update({"User-Agent": "TAFI-AI-Scanner/1.0"})


def api(path, params=None):
    r = session.get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def candles(symbol, interval, limit=100):
    raw = api("/fapi/v1/klines", {
        "symbol": symbol, "interval": interval, "limit": limit
    })
    df = pd.DataFrame(raw, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy", "taker_quote", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/period, adjust=False
    ).mean()
    loss = -delta.clip(upper=0).ewm(
        alpha=1/period, adjust=False
    ).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def analyze(df, direction):
    if len(df) < 30:
        return 0, {}

    c = df.close
    h = df.high
    l = df.low
    o = df.open
    v = df.volume

    # Gunakan candle yang sudah selesai
    i = -2
    price = float(c.iloc[i])
    prev = float(c.iloc[i-1])

    atr = float(
        pd.concat([
            h-l, (h-c.shift()).abs(),
            (l-c.shift()).abs()
        ], axis=1).max(axis=1).rolling(14).mean().iloc[i]
    )

    if not np.isfinite(atr) or atr <= 0:
        return 0, {}

    strength = 0
    reasons = []

    # Struktur swing
    swing_high = float(h.iloc[-22:-2].max())
    swing_low = float(l.iloc[-22:-2].min())
    midpoint = (swing_high + swing_low) / 2

    # Fibonacci retracement
    span = swing_high - swing_low
    fib382 = swing_high - span * 0.382
    fib618 = swing_high - span * 0.618

    # Candle rejection
    candle_range = max(
        float(h.iloc[i] - l.iloc[i]), 1e-12
    )
    upper_wick = float(
        h.iloc[i] - max(o.iloc[i], c.iloc[i])
    )
    lower_wick = float(
        min(o.iloc[i], c.iloc[i]) - l.iloc[i]
    )

    current_rsi = float(rsi(c).iloc[i])
    volume_ratio = float(
        v.iloc[i] / max(v.iloc[-22:-2].mean(), 1e-12)
    )

    if direction == "SHORT":
        if price < midpoint:
            strength += 15
            reasons.append("Struktur melemah")

        if price < swing_low:
            strength += 20
            reasons.append("Break support")

        if upper_wick / candle_range > 0.40:
            strength += 15
            reasons.append("Rejection atas")

        if current_rsi < 45:
            strength += 10
            reasons.append("Momentum bearish")

        if price <= fib382 and price >= fib618:
            strength += 10
            reasons.append("Zona Fibonacci")

        if volume_ratio > 1.3 and price < prev:
            strength += 15
            reasons.append("Volume jual meningkat")

        if c.iloc[i] < o.iloc[i]:
            strength += 5

    else:
        if price > midpoint:
            strength += 15
            reasons.append("Struktur menguat")

        if price > swing_high:
            strength += 20
            reasons.append("Break resistance")

        if lower_wick / candle_range > 0.40:
            strength += 15
            reasons.append("Rejection bawah")

        if current_rsi > 55:
            strength += 10
            reasons.append("Momentum bullish")

        fib_low = swing_low + span * 0.382
        fib_high = swing_low + span * 0.618

        if fib_low <= price <= fib_high:
            strength += 10
            reasons.append("Zona Fibonacci")

        if volume_ratio > 1.3 and price > prev:
            strength += 15
            reasons.append("Volume beli meningkat")

        if c.iloc[i] > o.iloc[i]:
            strength += 5

    return min(strength, 100), {
        "price": price,
        "atr": atr,
        "rsi": current_rsi,
        "volume_ratio": volume_ratio,
        "reasons": reasons
    }


def get_symbols():
    info = api("/fapi/v1/exchangeInfo")
    return [
        x["symbol"] for x in info["symbols"]
        if x["status"] == "TRADING"
        and x["contractType"] == "PERPETUAL"
        and x["quoteAsset"] == "USDT"
    ]


def get_tickers():
    data = api("/fapi/v1/ticker/24hr")
    return {x["symbol"]: float(x["quoteVolume"]) for x in data}


def fetch_4h(symbol):
    try:
        return symbol, candles(symbol, "4h", 100)
    except Exception:
        return symbol, None


def scan_market():
    symbols = get_symbols()
    tickers = get_tickers()

    print(f"\nTAFI: memindai {len(symbols)} kontrak Futures")

    results = []

    # Tahap 1: membaca struktur 4H semua koin
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_4h, s) for s in symbols]

        for future in as_completed(futures):
            symbol, df = future.result()
            if df is None:
                continue

            short_score, _ = analyze(df, "SHORT")
            long_score, _ = analyze(df, "LONG")

            best = max(short_score, long_score)

            if best >= 15:
                direction = (
                    "SHORT" if short_score >= long_score
                    else "LONG"
                )
                results.append((symbol, direction, best))

    # Prioritaskan kandidat struktur terkuat
    results.sort(key=lambda x: x[2], reverse=True)

    candidates = []

    # Tahap 2: konfirmasi 1H dan pemicu 15M
    for symbol, direction, score4h in results[:100]:
        try:
            df1 = candles(symbol, "1h", 100)
            df15 = candles(symbol, "15m", 60)

            score1h, data1 = analyze(df1, direction)
            score15, data15 = analyze(df15, direction)

            # Gabungan skor, bukan satu indikator
            total = (
                score4h * 0.40
                + score1h * 0.35
                + score15 * 0.25
            )

            if total < 45:
                continue

            entry = data15["price"]
            atr = data15["atr"]

            if direction == "SHORT":
                sl = entry + atr * 1.5
                tp1 = entry - atr * 2
                tp2 = entry - atr * 3
            else:
                sl = entry - atr * 1.5
                tp1 = entry + atr * 2
                tp2 = entry + atr * 3

            candidates.append({
                "symbol": symbol,
                "direction": direction,
                "score": round(total, 1),
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rsi": data1["rsi"],
                "volume": tickers.get(symbol, 0),
                "reasons": data1["reasons"] + data15["reasons"]
            })

        except Exception:
            continue

    candidates.sort(
        key=lambda x: (x["score"], x["volume"]),
        reverse=True
    )

    return candidates[:TOP_RESULTS]


def main():
    print("=" * 55)
    print("          TAFI AI FUTURES ROBOT")
    print("       LIVE MARKET INTELLIGENCE V1")
    print("=" * 55)

    while True:
        started = time.time()

        try:
            signals = scan_market()

            print("\n", datetime.now(timezone.utc))
            print("HASIL PEMINDAIAN:")

            if not signals:
                print("Belum ada kandidat memenuhi skor.")
            else:
                for x in signals:
                    print("\n-------------------------------")
                    print(x["symbol"], x["direction"])
                    print("SCORE:", x["score"], "/100")
                    print("ENTRY:", round(x["entry"], 8))
                    print("SL:", round(x["sl"], 8))
                    print("TP1:", round(x["tp1"], 8))
                    print("TP2:", round(x["tp2"], 8))
                    print("RSI 1H:", round(x["rsi"], 1))
                    print("ALASAN:", ", ".join(x["reasons"]))

            print("\nPemindaian selesai.")

        except Exception as e:
            print("Kesalahan koneksi:", str(e))

        elapsed = time.time() - started
        time.sleep(max(10, SCAN_INTERVAL - elapsed))


if __name__ == "__main__":
    main()
