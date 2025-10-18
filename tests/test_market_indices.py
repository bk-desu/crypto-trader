import numpy as np
import pandas as pd
import pytest

from features.market_indices import (
    SUPPORTED_INTERVALS,
    get_nasdaq_features,
    get_sp500_features,
)


def _fake_downloader_factory(close_values):
    idx = pd.date_range("2024-12-01", periods=len(close_values), freq="1D", tz="UTC")

    def _downloader(ticker, start, end):
        # Mimic yfinance's daily output: at least the Close column is required.
        data = pd.DataFrame({"Close": close_values}, index=idx)
        return data

    return _downloader


def _multiindex_downloader_factory(close_values):
    idx = pd.date_range("2024-12-01", periods=len(close_values), freq="1D", tz="UTC")

    def _downloader(ticker, start, end):
        data = pd.DataFrame({"Close": close_values, "Adj Close": close_values * 1.01}, index=idx)
        multi_cols = pd.MultiIndex.from_product(
            [["Close", "Adj Close"], [ticker]], names=["Price", "Ticker"]
        )
        return pd.DataFrame(data.values, index=idx, columns=multi_cols)

    return _downloader


@pytest.mark.parametrize("interval", ["1d", "4h", "1h"])
def test_sp500_features_structure(interval):
    downloader = _fake_downloader_factory(np.linspace(4000.0, 4050.0, num=40))
    start, end = "2025-01-05", "2025-01-10"

    df = get_sp500_features(start=start, end=end, interval=interval, downloader=downloader)

    expected_cols = {
        "sp500_close",
        "sp500_return",
        "sp500_rolling_mean_5",
        "sp500_rolling_mean_21",
        "sp500_rolling_std_21",
    }
    assert set(df.columns) == expected_cols
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp(start, tz="UTC")
    assert df.index[-1] == pd.Timestamp(end, tz="UTC")
    assert df.notna().all().all()


def test_nasdaq_features_reuses_helper():
    downloader = _fake_downloader_factory(np.linspace(12000.0, 12100.0, num=20))
    start, end = "2025-02-01", "2025-02-03"

    df = get_nasdaq_features(start=start, end=end, interval="4h", downloader=downloader)

    assert "nasdaq_close" in df.columns
    assert "nasdaq_return" in df.columns
    # Ensure frequency expansion to 4-hour grid
    expected_freq = pd.date_range(start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC"), freq="4h")
    assert len(df) == len(expected_freq)
    assert np.isfinite(df.iloc[0]["nasdaq_return"])


def test_invalid_interval_raises():
    downloader = _fake_downloader_factory(np.linspace(4000.0, 4010.0, num=5))
    with pytest.raises(ValueError):
        get_sp500_features("2025-01-01", "2025-01-03", interval="2d", downloader=downloader)

    with pytest.raises(ValueError):
        get_nasdaq_features("2025-01-01", "2025-01-03", interval="foo", downloader=downloader)


def test_sp500_features_handles_multiindex():
    close_vals = np.linspace(5000.0, 5010.0, num=130)
    downloader = _multiindex_downloader_factory(close_vals)
    start, end = "2025-03-01", "2025-03-05"

    df = get_sp500_features(start=start, end=end, interval="1d", downloader=downloader)

    assert set(df.columns) == {
        "sp500_close",
        "sp500_return",
        "sp500_rolling_mean_5",
        "sp500_rolling_mean_21",
        "sp500_rolling_std_21",
    }
    assert df.index[0] == pd.Timestamp(start, tz="UTC")
    assert np.isfinite(df["sp500_close"]).all()
