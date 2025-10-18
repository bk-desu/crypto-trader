"""Utilities for fetching equity index features (S&P 500, NASDAQ Composite).

The helpers return numeric feature frames aligned to the requested interval so they
can be joined directly onto the crypto feature matrix used by the HistoryManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf

__all__ = [
    "SUPPORTED_INTERVALS",
    "get_sp500_features",
    "get_nasdaq_features",
]

# Match the interval codes used by HistoryManager / backtest pipeline
_CODE_TO_PANDAS_FREQ = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}
SUPPORTED_INTERVALS = set(_CODE_TO_PANDAS_FREQ.keys())


Downloader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]


@dataclass(frozen=True)
class _IndexConfig:
    ticker: str
    prefix: str


_SP500_CFG = _IndexConfig(ticker="^GSPC", prefix="sp500")
_NASDAQ_CFG = _IndexConfig(ticker="^IXIC", prefix="nasdaq")

_BUFFER_DAYS = 90  # extra history to stabilise rolling statistics


def _default_downloader(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Wrapper around yfinance.download for easier mocking in tests."""

    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )


def _build_index_features(
    cfg: _IndexConfig,
    start: str,
    end: str,
    interval: Literal[
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    ] = "1d",
    *,
    downloader: Optional[Downloader] = None,
) -> pd.DataFrame:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(SUPPORTED_INTERVALS)}")

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    # add a buffer so rolling windows have enough history
    start_buffer = (start_ts - pd.Timedelta(days=_BUFFER_DAYS)).tz_localize(None)
    end_buffer = (end_ts + pd.Timedelta(days=1)).tz_localize(None)

    dl = downloader or _default_downloader
    raw = dl(cfg.ticker, start_buffer, end_buffer)
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                f"{cfg.prefix}_close",
                f"{cfg.prefix}_return",
                f"{cfg.prefix}_rolling_mean_5",
                f"{cfg.prefix}_rolling_mean_21",
                f"{cfg.prefix}_rolling_std_21",
            ]
        )

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >=0.2 returns a MultiIndex of (Price, Ticker) for single tickers.
        try:
            if cfg.ticker in df.columns.get_level_values(-1):
                df = df.xs(cfg.ticker, axis=1, level=-1)
        except (KeyError, IndexError):
            pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(part) for part in col if part and str(part) != "nan")
                for col in df.columns
            ]

    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df.loc[(df.index >= start_ts - pd.Timedelta(days=_BUFFER_DAYS)) & (df.index <= end_ts)]

    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if close_col is None:
        raise ValueError("Downloaded index frame does not contain a Close column")

    close = pd.to_numeric(df[close_col], errors="coerce")
    feat = pd.DataFrame(
        {
            f"{cfg.prefix}_close": close,
            f"{cfg.prefix}_return": close.pct_change().fillna(0.0),
            f"{cfg.prefix}_rolling_mean_5": close.rolling(window=5, min_periods=1).mean(),
            f"{cfg.prefix}_rolling_mean_21": close.rolling(window=21, min_periods=1).mean(),
            f"{cfg.prefix}_rolling_std_21": close.pct_change().rolling(window=21, min_periods=1).std().fillna(0.0),
        },
        index=close.index,
    )

    feat = feat.dropna(how="all")
    if feat.empty:
        return feat

    target_index = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=_CODE_TO_PANDAS_FREQ[interval],
        tz="UTC",
    )
    feat = feat.reindex(target_index, method="ffill")
    feat.index.name = "ts_utc"
    feat = feat.ffill().bfill()

    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    return feat[numeric_cols]


def get_sp500_features(
    start: str,
    end: str,
    interval: Literal[
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    ] = "1d",
    *,
    downloader: Optional[Downloader] = None,
) -> pd.DataFrame:
    """Return aligned S&P 500 (^GSPC) features for the requested interval."""

    return _build_index_features(_SP500_CFG, start=start, end=end, interval=interval, downloader=downloader)


def get_nasdaq_features(
    start: str,
    end: str,
    interval: Literal[
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    ] = "1d",
    *,
    downloader: Optional[Downloader] = None,
) -> pd.DataFrame:
    """Return aligned NASDAQ Composite (^IXIC) features for the requested interval."""

    return _build_index_features(_NASDAQ_CFG, start=start, end=end, interval=interval, downloader=downloader)
