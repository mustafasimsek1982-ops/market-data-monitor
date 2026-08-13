from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scanner import active_usdt_symbols, due_intervals, evaluate, ichimoku


def frame(size=160):
    x = np.arange(size, dtype=float)
    close = 100 + x * 0.1
    return pd.DataFrame({"open_time": x, "close_time": x, "open": close,
                         "high": close + 2, "low": close - 2, "close": close})


def test_ichimoku_cloud_is_shifted_26_bars():
    df = frame()
    i = ichimoku(df)
    assert pd.isna(i.span_b.iloc[76])
    assert not pd.isna(i.span_b.iloc[77])


def test_partial_signal_is_never_a_match(monkeypatch):
    df = frame()
    monkeypatch.setattr("scanner.luxalgo_green_b", lambda _: pd.Series([False] * (len(df) - 1) + [True]))
    assert evaluate(df, "1h")["matched"] is False


def test_due_intervals_follow_binance_utc_closes():
    assert due_intervals(datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)) == ["15m", "1h", "4h"]
    assert due_intervals(datetime(2026, 1, 1, 8, 2, tzinfo=timezone.utc)) == ["15m", "1h", "4h"]
    assert due_intervals(datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)) == ["15m"]


class FakeResponse:
    @staticmethod
    def raise_for_status():
        return None

    def json(self):
        return {"symbols": [
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
            {"symbol": "BTCUPUSDT", "baseAsset": "BTCUP", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
            {"symbol": "USDCUSDT", "baseAsset": "USDC", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
        ]}


class FakeSession:
    @staticmethod
    def get(*args, **kwargs):
        return FakeResponse()


def test_symbol_filter():
    assert active_usdt_symbols(FakeSession) == ["BTCUSDT"]
