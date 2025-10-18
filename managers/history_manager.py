# Provides HistoricalData, a thin OOP wrapper around Binance klines that:
# - Loads historical OHLCV
# - Computes indicators (EMA, CMO, +DM, -DM) and candlestick patterns
# - Builds a clean features DataFrame with a next-bar target (UP_DOWN)
# - Adds forward-return targets: FWD_RET, FWD_RET_BPS
# - Merges Fear & Greed and BTC on-chain features (defaults: enabled)
# - Updates with the latest closed bar and exposes prediction features

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
from binance.client import Client

from features.fear_index import get_fng_features
from features.fomc import get_fomc_sentiment
from features.market_indices import get_nasdaq_features, get_sp500_features
from features.on_chain_data import get_btc_onchain_smoothed

INTERVAL_TO_MS = {
    Client.KLINE_INTERVAL_1MINUTE: 60_000,
    Client.KLINE_INTERVAL_3MINUTE: 3 * 60_000,
    Client.KLINE_INTERVAL_5MINUTE: 5 * 60_000,
    Client.KLINE_INTERVAL_15MINUTE: 15 * 60_000,
    Client.KLINE_INTERVAL_30MINUTE: 30 * 60_000,
    Client.KLINE_INTERVAL_1HOUR: 60 * 60_000,
    Client.KLINE_INTERVAL_2HOUR: 2 * 60 * 60_000,
    Client.KLINE_INTERVAL_4HOUR: 4 * 60 * 60_000,
    Client.KLINE_INTERVAL_6HOUR: 6 * 60 * 60_000,
    Client.KLINE_INTERVAL_8HOUR: 8 * 60 * 60_000,
    Client.KLINE_INTERVAL_12HOUR: 12 * 60 * 60_000,
    Client.KLINE_INTERVAL_1DAY: 24 * 60 * 60_000,
}

# Map Binance constants to the code strings expected by the feature modules
_BINANCE_CONST_TO_CODE = {
    Client.KLINE_INTERVAL_1MINUTE: "1m",
    Client.KLINE_INTERVAL_3MINUTE: "3m",
    Client.KLINE_INTERVAL_5MINUTE: "5m",
    Client.KLINE_INTERVAL_15MINUTE: "15m",
    Client.KLINE_INTERVAL_30MINUTE: "30m",
    Client.KLINE_INTERVAL_1HOUR: "1h",
    Client.KLINE_INTERVAL_2HOUR: "2h",
    Client.KLINE_INTERVAL_4HOUR: "4h",
    Client.KLINE_INTERVAL_6HOUR: "6h",
    Client.KLINE_INTERVAL_8HOUR: "8h",
    Client.KLINE_INTERVAL_12HOUR: "12h",
    Client.KLINE_INTERVAL_1DAY: "1d",
}


def _as_dataframe(klines: list) -> pd.DataFrame:
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open_time", "close_time", "number_of_trades"]:
        df[c] = pd.to_numeric(df[c], downcast="integer", errors="coerce")
    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def _bin_pattern_to_binary(arr: np.ndarray) -> np.ndarray:
    return (np.sign(arr) > 0).astype(np.int32)


class HistoryManager:
    """
    Manages historical and latest market data, indicators, and features.
    """

    def __init__(
        self,
        client: Client,
        symbol: str,
        interval: str,
        start_str: str,
        timelag: int = 20,
        *,
        # external features (defaults ON as requested)
        include_fng: bool = True,
        fng_kwargs: Optional[Dict[str, Any]] = None,
        include_onchain: bool = True,
        onchain_kwargs: Optional[Dict[str, Any]] = None,
        include_fomc: bool = True,
        fomc_kwargs: Optional[Dict[str, Any]] = None,
        include_sp500: bool = True,
        sp500_kwargs: Optional[Dict[str, Any]] = None,
        include_nasdaq: bool = True,
        nasdaq_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.timelag = timelag

        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {interval}")

        self.interval_ms = INTERVAL_TO_MS[interval]

        self.include_fng = include_fng
        self.fng_kwargs = fng_kwargs or {}
        self.include_onchain = include_onchain
        self.onchain_kwargs = onchain_kwargs or {}
        self.include_fomc = include_fomc
        self.fomc_kwargs = fomc_kwargs or {}
        self.include_sp500 = include_sp500
        self.sp500_kwargs = sp500_kwargs or {}
        self.include_nasdaq = include_nasdaq
        self.nasdaq_kwargs = nasdaq_kwargs or {}

        # data holders
        self.df_ohlcv: pd.DataFrame = pd.DataFrame()
        self.df_features: pd.DataFrame = pd.DataFrame()

        # base predictors (we auto-extend if external numeric columns are added)
        self.predictor_cols = [
            "EMA",
            "CMO",
            "MINUSDM",
            "PLUSDM",
            "CLOSE",
            "CLOSEL1",
            "CLOSEL2",
            "CLOSEL3",
            "PATT_3OUT",
            "PATT_CMB",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "MACD_HIST",
            "ADX",
            "ATR",
            "NATR",
            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",
            "BB_WIDTH",
            "OBV",
            "MFI",
            "AD",
            "ADOSC",
            "STOCH_K",
            "STOCH_D",
            "TRIX",
            "ROC",
        ]

    # ---------- Data Fetch ----------

    def load_history(self) -> None:
        klines = self.client.get_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            start_str=self.start_str,
        )
        self.df_ohlcv = _as_dataframe(klines)
        if self.df_ohlcv.empty:
            raise RuntimeError("No historical data returned.")
        self._recompute_features()

    def update_with_latest(self, limit: int = 2) -> None:
        recent = self.client.get_klines(
            symbol=self.symbol, interval=self.interval, limit=limit
        )
        df_recent = _as_dataframe(recent)

        if self.df_ohlcv.empty:
            self.df_ohlcv = df_recent.copy()
        else:
            last_open_time = int(self.df_ohlcv["open_time"].iloc[-1])
            new_rows = df_recent[df_recent["open_time"] > last_open_time]
            if not new_rows.empty:
                self.df_ohlcv = pd.concat([self.df_ohlcv, new_rows], ignore_index=True)

        self._recompute_features()

    # ---------- Features ----------

    def _recompute_features(self) -> None:
        df = self.df_ohlcv.copy()

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        openp = df["open"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)

        ema = talib.EMA(close, timeperiod=self.timelag)
        cmo = talib.CMO(close, timeperiod=self.timelag)
        minusdm = talib.MINUS_DM(high, low, timeperiod=self.timelag)
        plusdm = talib.PLUS_DM(high, low, timeperiod=self.timelag)

        patt_3out = _bin_pattern_to_binary(talib.CDL3OUTSIDE(openp, high, low, close))
        patt_cmb = _bin_pattern_to_binary(
            talib.CDLCLOSINGMARUBOZU(openp, high, low, close)
        )

        closel1 = df["close"].shift(1)
        closel2 = df["close"].shift(2)
        closel3 = df["close"].shift(3)

        rsi = talib.RSI(close, timeperiod=self.timelag)
        macd, macd_sig, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        adx = talib.ADX(high, low, close, timeperiod=self.timelag)
        trix = talib.TRIX(close, timeperiod=self.timelag)
        roc = talib.ROC(close, timeperiod=self.timelag)

        atr = talib.ATR(high, low, close, timeperiod=self.timelag)
        natr = talib.NATR(high, low, close, timeperiod=self.timelag)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=self.timelag, nbdevup=2, nbdevdn=2, matype=0
        )
        bb_width = np.divide(
            bb_upper - bb_lower,
            bb_middle,
            out=np.full_like(bb_middle, np.nan, dtype=float),
            where=(bb_middle != 0) & ~np.isnan(bb_middle),
        )

        obv = talib.OBV(close, vol)
        mfi = talib.MFI(high, low, close, vol, timeperiod=self.timelag)
        ad = talib.AD(high, low, close, vol)
        adosc = talib.ADOSC(high, low, close, vol, fastperiod=3, slowperiod=10)

        k_period = max(int(self.timelag), 5)
        d_period = 3
        stoch_k, stoch_d = talib.STOCH(
            high,
            low,
            close,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0,
        )

        feat = pd.DataFrame(
            {
                "EMA": ema,
                "CMO": cmo,
                "MINUSDM": minusdm,
                "PLUSDM": plusdm,
                "CLOSE": df["close"],
                "CLOSEL1": closel1,
                "CLOSEL2": closel2,
                "CLOSEL3": closel3,
                "PATT_3OUT": patt_3out,
                "PATT_CMB": patt_cmb,
                "RSI": rsi,
                "MACD": macd,
                "MACD_SIGNAL": macd_sig,
                "MACD_HIST": macd_hist,
                "ADX": adx,
                "ATR": atr,
                "NATR": natr,
                "BB_UPPER": bb_upper,
                "BB_MIDDLE": bb_middle,
                "BB_LOWER": bb_lower,
                "BB_WIDTH": bb_width,
                "OBV": obv,
                "MFI": mfi,
                "AD": ad,
                "ADOSC": adosc,
                "STOCH_K": stoch_k,
                "STOCH_D": stoch_d,
                "TRIX": trix,
                "ROC": roc,
            },
            index=df.index,
        )

        # (NEW) forward returns (direction + magnitude)
        up_down = (df["close"].shift(-1) > df["close"]).astype("float64")
        feat["UP_DOWN"] = up_down
        feat["FWD_RET"] = (df["close"].shift(-1) / df["close"]) - 1.0
        feat["FWD_RET_BPS"] = 1e4 * feat["FWD_RET"]

        # (NEW) external features
        ts_utc = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        start_iso = ts_utc.iloc[0].isoformat()
        end_iso = ts_utc.iloc[-1].isoformat()
        code_iv = _BINANCE_CONST_TO_CODE[self.interval]

        base_cols = set(feat.columns)
        appended_cols: List[str] = []

        # Fear & Greed — keep numeric only (drop the string label)
        if self.include_fng:
            try:
                fng_df = get_fng_features(
                    start=start_iso, end=end_iso, interval=code_iv, **self.fng_kwargs
                )
                if not fng_df.empty:
                    fng_num = fng_df.select_dtypes(include=[np.number])
                    if not fng_num.empty:
                        fng_aligned = fng_num.reindex(ts_utc, method="ffill")
                        fng_aligned.index = feat.index
                        feat = feat.join(fng_aligned, how="left")
                        appended_cols.extend(list(fng_aligned.columns))
            except Exception:
                pass

        # fomc sentiment
        if self.include_fomc:
            try:
                fomc_df = get_fomc_sentiment(
                    start=start_iso,
                    end=end_iso,
                    interval=code_iv,
                    **self.fomc_kwargs,
                )
                if not fomc_df.empty:
                    fomc_num = fomc_df.select_dtypes(include=[np.number])
                    if not fomc_num.empty:
                        fomc_aligned = fomc_num.reindex(ts_utc, method="ffill")
                        fomc_aligned.index = feat.index
                        feat = feat.join(fomc_aligned, how="left")
                        appended_cols.extend(list(fomc_aligned.columns))
            except Exception:
                pass

        # On-chain — numeric frame already
        if self.include_onchain:
            try:
                oc_df = get_btc_onchain_smoothed(
                    start=start_iso,
                    end=end_iso,
                    interval=code_iv,
                    **self.onchain_kwargs,
                )
                if not oc_df.empty:
                    oc_num = oc_df.select_dtypes(include=[np.number])
                    if not oc_num.empty:
                        oc_aligned = oc_num.reindex(ts_utc, method="ffill")
                        oc_aligned.index = feat.index
                        feat = feat.join(oc_aligned, how="left")
                        appended_cols.extend(list(oc_aligned.columns))
            except Exception:
                pass

        # S&P 500 index
        if self.include_sp500:
            try:
                sp_df = get_sp500_features(
                    start=start_iso,
                    end=end_iso,
                    interval=code_iv,
                    **self.sp500_kwargs,
                )
                if not sp_df.empty:
                    sp_num = sp_df.select_dtypes(include=[np.number])
                    if not sp_num.empty:
                        sp_aligned = sp_num.reindex(ts_utc, method="ffill")
                        sp_aligned.index = feat.index
                        feat = feat.join(sp_aligned, how="left")
                        appended_cols.extend(list(sp_aligned.columns))
            except Exception:
                pass

        # NASDAQ Composite index
        if self.include_nasdaq:
            try:
                nd_df = get_nasdaq_features(
                    start=start_iso,
                    end=end_iso,
                    interval=code_iv,
                    **self.nasdaq_kwargs,
                )
                if not nd_df.empty:
                    nd_num = nd_df.select_dtypes(include=[np.number])
                    if not nd_num.empty:
                        nd_aligned = nd_num.reindex(ts_utc, method="ffill")
                        nd_aligned.index = feat.index
                        feat = feat.join(nd_aligned, how="left")
                        appended_cols.extend(list(nd_aligned.columns))
            except Exception:
                pass

        # numeric-only safety for models
        non_numeric = feat.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            feat = feat.drop(columns=non_numeric)

        # finalize target dtype
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        feat = feat.astype({"UP_DOWN": "int64"})

        # extend predictor list with any newly joined numeric columns (keep order)
        if appended_cols:
            for c in feat.columns:
                if c not in base_cols and c not in (
                    "UP_DOWN",
                    "FWD_RET",
                    "FWD_RET_BPS",
                ):
                    if c not in self.predictor_cols and pd.api.types.is_numeric_dtype(
                        feat[c]
                    ):
                        self.predictor_cols.append(c)

        self.df_features = feat

    # ---------- Accessors ----------

    def last_closed_open_time_ms(self) -> int:
        last_idx = self.df_features.index[-1]
        return int(self.df_ohlcv.loc[last_idx, "open_time"])

    def next_bar_close_time_ms(self) -> int:
        return self.last_closed_open_time_ms() + self.interval_ms

    def dataset(self, target: str = "direction") -> Tuple[pd.DataFrame, pd.Series]:
        """
        target:
          - 'direction'   -> UP_DOWN (int)
          - 'return'      -> FWD_RET (float)
          - 'return_bps'  -> FWD_RET_BPS (float)
        """
        X = self.df_features[self.predictor_cols].copy()
        if target == "direction":
            y = self.df_features["UP_DOWN"].copy()
        elif target == "return":
            y = self.df_features["FWD_RET"].copy()
        elif target == "return_bps":
            y = self.df_features["FWD_RET_BPS"].copy()
        else:
            raise ValueError("target must be 'direction'|'return'|'return_bps'")
        m = y.notna()
        return X.loc[m], y.loc[m]

    def latest_features_row(self) -> pd.DataFrame:
        return self.df_features[self.predictor_cols].iloc[[-1]].copy()
