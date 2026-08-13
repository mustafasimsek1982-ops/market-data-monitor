from datetime import datetime
from zoneinfo import ZoneInfo

from bist_scanner import due_intervals


TR = ZoneInfo("Europe/Istanbul")


def test_due_intervals_at_14_close():
    assert due_intervals(datetime(2026, 8, 13, 14, 2, tzinfo=TR)) == ["15m", "1h", "4h"]


def test_due_intervals_at_quarter_hour():
    assert due_intervals(datetime(2026, 8, 13, 10, 17, tzinfo=TR)) == ["15m"]


def test_due_intervals_weekend():
    assert due_intervals(datetime(2026, 8, 15, 14, 2, tzinfo=TR)) == []
