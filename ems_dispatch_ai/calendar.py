"""Saudi calendar effects (Gregorian dates of Islamic periods; Umm al-Qura based, +-1 day)."""
from __future__ import annotations

from datetime import date

RAMADAN = [(date(2024, 3, 11), date(2024, 4, 9)), (date(2025, 3, 1), date(2025, 3, 29)),
           (date(2026, 2, 18), date(2026, 3, 19))]
EID_AL_FITR = [(date(2024, 4, 10), date(2024, 4, 13)), (date(2025, 3, 30), date(2025, 4, 2)),
               (date(2026, 3, 20), date(2026, 3, 23))]
HAJJ = [(date(2024, 6, 13), date(2024, 6, 19)), (date(2025, 6, 3), date(2025, 6, 9)),
        (date(2026, 5, 24), date(2026, 5, 30))]  # 8-13 Dhu al-Hijjah incl. Eid al-Adha


def _in(d: date, periods) -> bool:
    return any(a <= d <= b for a, b in periods)


def is_ramadan(d: date) -> bool:
    return _in(d, RAMADAN)


def is_eid(d: date) -> bool:
    return _in(d, EID_AL_FITR)


def is_hajj(d: date) -> bool:
    return _in(d, HAJJ)


def is_weekend(d: date) -> bool:
    return d.weekday() in (4, 5)  # Friday, Saturday in Saudi Arabia
