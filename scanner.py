import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Binance'in yalnızca herkese açık piyasa verileri için önerdiği resmi adres.
BASE_URL = "https://data-api.binance.vision"
INTERVAL_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "TRY", "BUSD"}
LEVERAGED_ENDINGS = ("UP", "DOWN", "BULL", "BEAR")


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # TradingView'da mevcut mumun altında görülen bulut, 26 mum önce hesaplanan span'lardır.
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b})


def pine_rma(values: pd.Series, length: int) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) < length:
        return out
    out.iloc[length - 1] = values.iloc[:length].mean()
    for i in range(length, len(values)):
        out.iloc[i] = (out.iloc[i - 1] * (length - 1) + values.iloc[i]) / length
    return out


def luxalgo_green_b(df: pd.DataFrame, length: int = 14, mult: float = 1.0) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    slope = pine_rma(tr, length) / length * mult
    upper, slope_ph, upos = 0.0, 0.0, 0
    signals = pd.Series(False, index=df.index)

    for i in range(len(df)):
        ph = np.nan
        pivot = i - length
        if i >= length * 2:
            window = high.iloc[pivot - length:pivot + length + 1]
            candidate = high.iloc[pivot]
            if candidate == window.max():
                ph = candidate
        previous_upos = upos
        if not np.isnan(ph):
            slope_ph = slope.iloc[i] if not np.isnan(slope.iloc[i]) else slope_ph
            upper = ph
            upos = 0
        else:
            upper -= slope_ph
            if close.iloc[i] > upper - slope_ph * length:
                upos = 1
        signals.iloc[i] = upos > previous_upos
    return signals


def evaluate(df: pd.DataFrame, interval: str) -> dict:
    close = df["close"]
    ma7, ma25, ma99 = sma(close, 7), sma(close, 25), sma(close, 99)
    ma_cross = bool(ma7.iloc[-1] > ma25.iloc[-1] and ma7.iloc[-1] > ma99.iloc[-1]
                    and (ma7.iloc[-2] <= ma25.iloc[-2] or ma7.iloc[-2] <= ma99.iloc[-2]))
    lux_b = bool(luxalgo_green_b(df).iloc[-1])
    result = {"ma_cross": ma_cross, "luxalgo_green_b": lux_b}
    if interval in {"1h", "4h"}:
        ichi = ichimoku(df)
        tenkan_cross = bool(ichi.tenkan.iloc[-1] > ichi.kijun.iloc[-1]
                            and ichi.tenkan.iloc[-2] <= ichi.kijun.iloc[-2])
        cloud_top = max(ichi.span_a.iloc[-1], ichi.span_b.iloc[-1])
        above_cloud = bool(close.iloc[-1] > cloud_top)
        result.update({"tenkan_kijun_cross": tenkan_cross, "above_cloud": above_cloud})
    result["matched"] = all(result.values())
    return result


def active_usdt_symbols(session=requests) -> list[str]:
    response = session.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=20)
    response.raise_for_status()
    data = response.json()
    if "symbols" not in data:
        raise RuntimeError(f"Binance exchangeInfo hatası: {data}")
    symbols = []
    for item in data["symbols"]:
        base = item["baseAsset"]
        if (item["quoteAsset"] == "USDT" and item["status"] == "TRADING"
                and item.get("isSpotTradingAllowed", True) and base not in STABLE_BASES
                and not base.endswith(LEVERAGED_ENDINGS)):
            symbols.append(item["symbol"])
    return symbols


def closed_klines(symbol: str, interval: str, session=requests, now_ms: int | None = None) -> pd.DataFrame:
    response = session.get(f"{BASE_URL}/api/v3/klines",
                           params={"symbol": symbol, "interval": interval, "limit": 500}, timeout=20)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"Binance kline hatası ({symbol}/{interval}): {rows}")
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    current = now_ms if now_ms is not None else int(time.time() * 1000)
    return df[df.close_time.astype("int64") < current].reset_index(drop=True)


def due_intervals(now: datetime) -> list[str]:
    minute = now.minute
    # GitHub zamanlayıcısı birkaç dakika gecikebilir; ilk 7 dakikalık pencereyi kabul et.
    due = ["15m"] if minute % 15 <= 7 else []
    if minute <= 7:
        due.append("1h")
        if now.hour % 4 == 0:
            due.append("4h")
    return due


def send_telegram(text: str, session=requests) -> None:
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    response = session.post(f"https://api.telegram.org/bot{token}/sendMessage",
                            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True}, timeout=20)
    response.raise_for_status()


def load_state(path: Path) -> set[str]:
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def main() -> None:
    if os.getenv("MANUAL_TEST", "").lower() == "true":
        send_telegram("✅ Market Data Monitor bağlantı testi başarılı. Otomatik Binance taraması etkin.")
        return

    now = datetime.now(timezone.utc)
    intervals = due_intervals(now)
    if not intervals:
        return
    state_path = Path(os.getenv("SIGNAL_STATE_FILE", ".signal-state.json"))
    sent = load_state(state_path)
    def analyze(symbol: str, interval: str):
        df = closed_klines(symbol, interval)
        return symbol, interval, df, evaluate(df, interval)

    candidates = [(symbol, interval) for symbol in active_usdt_symbols() for interval in intervals]
    workers = min(int(os.getenv("MAX_WORKERS", "12")), len(candidates))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze, symbol, interval): (symbol, interval)
                   for symbol, interval in candidates}
        for future in as_completed(futures):
            symbol, interval = futures[future]
            try:
                symbol, interval, df, checks = future.result()
                candle = int(df.open_time.iloc[-1])
                key = f"{symbol}:{interval}:{candle}"
                if checks["matched"] and key not in sent:
                    price = df.close.iloc[-1]
                    labels = ["MA7, MA25 ve MA99 yukarı kesişimi", "LuxAlgo yeşil B"]
                    if interval in {"1h", "4h"}:
                        labels += ["Tenkan, Kijun'u yukarı kesti", "Fiyat Ichimoku bulutu üzerinde"]
                    message = (f"🚨 BINANCE USDT SİNYALİ\n\n{symbol} • {interval}\nFiyat: {price:g}\n"
                               + "\n".join(f"✅ {x}" for x in labels)
                               + f"\n\nhttps://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}")
                    send_telegram(message)
                    sent.add(key)
            except Exception as exc:
                print(f"{symbol} {interval}: {exc}")
    # Eski kayıtların sınırsız büyümesini önle.
    state_path.write_text(json.dumps(sorted(sent)[-5000:]), encoding="utf-8")


if __name__ == "__main__":
    main()
