from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from scanner import evaluate, send_telegram


TV_SCAN_URL = "https://scanner.tradingview.com/turkey/scan"
ISTANBUL = ZoneInfo("Europe/Istanbul")
STATE_FILE = Path(os.getenv("BIST_SIGNAL_STATE_FILE", ".bist-signal-state.json"))


def get_bist_symbols(session=requests) -> list[str]:
    """TradingView tarayıcısından aktif BIST adi hisse kodlarını al."""
    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "BIST"},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "options": {"lang": "tr"},
        "markets": ["turkey"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "description", "type", "subtype"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 2000],
    }
    response = session.post(TV_SCAN_URL, json=payload, timeout=30)
    response.raise_for_status()
    rows = response.json().get("data", [])
    symbols = []
    for row in rows:
        values = row.get("d", [])
        if not values:
            continue
        symbol = str(values[0]).strip().upper()
        subtype = str(values[3]).lower() if len(values) > 3 else ""
        if symbol and subtype not in {"preferred", "etf", "fund"}:
            symbols.append(symbol)
    result = sorted(set(symbols))
    if len(result) < 100:
        raise RuntimeError(f"BIST hisse listesi yalnızca {len(result)} kod döndürdü")
    return result


def _localize_index(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.index.tz is None:
        data.index = data.index.tz_localize(ISTANBUL)
    else:
        data.index = data.index.tz_convert(ISTANBUL)
    return data


def completed_session_bars(
    frame: pd.DataFrame, interval: str, now: datetime | pd.Timestamp | None = None
) -> pd.DataFrame:
    columns = ["Open", "High", "Low", "Close", "Volume"]
    data = frame[columns].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if data.empty:
        return data
    data = _localize_index(data)
    local_now = pd.Timestamp(now or datetime.now(ISTANBUL))
    if local_now.tzinfo is None:
        local_now = local_now.tz_localize(ISTANBUL)
    else:
        local_now = local_now.tz_convert(ISTANBUL)

    session = (
        (data.index.dayofweek < 5)
        & (data.index.time >= pd.Timestamp("10:00").time())
        & (data.index.time < pd.Timestamp("18:00").time())
    )
    data = data[session]
    if interval == "15m":
        return data[data.index + pd.Timedelta(minutes=15) <= local_now]
    if interval == "1h":
        return data[data.index + pd.Timedelta(hours=1) <= local_now]
    if interval != "4h":
        raise ValueError(f"Desteklenmeyen aralık: {interval}")

    bars = data.resample(
        "4h", origin="start_day", offset="10h", label="left", closed="left"
    ).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    bars = bars[bars.index.hour.isin([10, 14])]
    return bars[bars.index + pd.Timedelta(hours=4) <= local_now]


def download_batch(symbols: list[str], interval: str) -> dict[str, pd.DataFrame]:
    tickers = [f"{symbol}.IS" for symbol in symbols]
    yahoo_interval = "15m" if interval == "15m" else "1h"
    period = "60d" if interval == "15m" else "2y"
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=yahoo_interval,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=30,
    )
    frames: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        frame = completed_session_bars(raw, interval)
        if not frame.empty:
            frames[symbols[0]] = frame
        return frames
    for symbol, ticker in zip(symbols, tickers):
        try:
            frame = completed_session_bars(raw[ticker].dropna(how="all"), interval)
            if not frame.empty:
                frames[symbol] = frame
        except (KeyError, ValueError):
            continue
    return frames


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})


def due_intervals(now: datetime | None = None) -> list[str]:
    local = (now or datetime.now(ISTANBUL)).astimezone(ISTANBUL)
    if local.weekday() >= 5:
        return []
    result = []
    minute = local.minute
    clock = local.hour * 60 + minute
    if 10 * 60 + 15 <= clock <= 18 * 60 + 7 and minute % 15 <= 7:
        result.append("15m")
    if 11 <= local.hour <= 18 and minute <= 7:
        result.append("1h")
    if local.hour in {14, 18} and minute <= 7:
        result.append("4h")
    return result


def load_state() -> set[str]:
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def signal_message(symbol: str, interval: str, price: float) -> str:
    labels = ["MA7, MA25 ve MA99 yukarı kesişimi", "LuxAlgo yeşil B"]
    if interval in {"1h", "4h"}:
        labels += ["Tenkan, Kijun'u yukarı kesti", "Fiyat Ichimoku bulutu üzerinde"]
    return (
        f"🚨 BIST SİNYALİ\n\n{symbol} • {interval}\nFiyat: {price:g} TL\n"
        + "\n".join(f"✅ {label}" for label in labels)
        + f"\n\nhttps://www.tradingview.com/chart/?symbol=BIST%3A{symbol}"
        + "\n\nYatırım tavsiyesi değildir."
    )


def main() -> None:
    if os.getenv("MANUAL_TEST", "").lower() == "true":
        send_telegram("✅ Market Data Monitor bağlantı testi başarılı. Otomatik BIST taraması etkin.")
        return

    intervals = due_intervals()
    if not intervals:
        return
    symbols = get_bist_symbols()
    sent = load_state()
    batch_size = max(10, int(os.getenv("BIST_BATCH_SIZE", "60")))
    for interval in intervals:
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            try:
                frames = download_batch(batch, interval)
            except Exception as exc:
                print(f"{interval} grup {start}: {exc}")
                continue
            for symbol, frame in frames.items():
                try:
                    data = normalize(frame)
                    if len(data) < 110:
                        continue
                    checks = evaluate(data, interval)
                    candle = pd.Timestamp(data.index[-1]).isoformat()
                    key = f"{symbol}:{interval}:{candle}"
                    if checks["matched"] and key not in sent:
                        send_telegram(signal_message(symbol, interval, float(data.close.iloc[-1])))
                        sent.add(key)
                except Exception as exc:
                    print(f"{symbol} {interval}: {exc}")
            time.sleep(1)
    STATE_FILE.write_text(json.dumps(sorted(sent)[-5000:], ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
