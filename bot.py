import math
import time

import numpy as np
import pandas as pd
from binance.client import Client

import config
from evaluations.evaluator import choose_best_threshold_for_window
from evaluations.metrics import SweepConfig
from managers.history_manager import INTERVAL_TO_MS, HistoryManager
from managers.model_manager import ModelManager
from managers.position_manager import PositionManager, SizingConfig

_INTERVAL_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
}


def _map_trade_interval_to_binance(code: str) -> str:
    if code not in _INTERVAL_MAP:
        raise ValueError(f"Unsupported trade_interval: {code}")
    return _INTERVAL_MAP[code]


def _make_rolling_windows(
    df: pd.DataFrame, feat_cols: list[str], window: int
) -> np.ndarray:
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    if len(arr) < window:
        return np.empty(
            (0, window, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32
        )
    n = len(arr) - window + 1
    out = np.empty((n, window, arr.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = arr[i : i + window]
    return out


def _last_window(df: pd.DataFrame, feat_cols: list[str], window: int) -> np.ndarray:
    if len(df) < window:
        raise RuntimeError(f"Not enough rows ({len(df)}) for window={window}")
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    return arr[-window:].reshape(1, window, arr.shape[1])


def _logistic_from_bps(bps: float, scale_bps: float = 20.0) -> float:
    """
    Convert a predicted return in bps into a pseudo-probability for Kelly sizing.
    scale_bps controls steepness; default 20 bps ~ mid sensitivity.
    """
    x = float(bps) / max(1e-9, float(scale_bps))
    return 1.0 / (1.0 + math.exp(-x))


def _sweep_on_predicted_return(
    yhat_bps: np.ndarray,
    fwd_ret_bps: np.ndarray,
    cost_bps: float,
    best_metric: str,
    thr_grid_bps: np.ndarray | None = None,
) -> dict:
    """
    Long-only: trade when predicted return >= (threshold + cost).
    Metrics computed net of costs (bps).
    """
    if thr_grid_bps is None:
        thr_grid_bps = np.arange(0.0, 60.1, 1.0)  # 0..60 bps

    best = {"threshold": 0.0, "total_net_return": float("-inf")}
    for thr in thr_grid_bps:
        take = yhat_bps >= (thr + cost_bps)
        trades = int(take.sum())
        if trades == 0:
            continue
        net = fwd_ret_bps[take] - cost_bps
        total = float(np.nansum(net))
        avg = float(np.nanmean(net))
        std = float(np.nanstd(net))
        sharpe_like = float(avg / (std + 1e-9))
        metric_val = {
            "total_net_return": total,
            "avg_net_ret_per_bar": avg,
            "sharpe_like": sharpe_like,
        }[best_metric]
        if metric_val > best.get(best_metric, float("-inf")):
            best = {
                "threshold": float(thr),
                "trades": trades,
                "total_net_return": total,
                "avg_net_ret_per_bar": avg,
                "sharpe_like": sharpe_like,
            }
    if best["total_net_return"] == float("-inf"):
        best.update(
            {
                "threshold": 0.0,
                "trades": 0,
                "total_net_return": 0.0,
                "avg_net_ret_per_bar": float("nan"),
                "sharpe_like": float("nan"),
            }
        )
    return best


class TradeBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        trade_interval_code: str | None = None,
        start_str: str | None = None,
        timelag: int = 20,
        retrain_every: int = 50,
        sweep_cfg: SweepConfig | None = None,
        best_metric: str | None = None,
        calib_window_bars: int = 2000,
        fees_bps: float | None = None,
        slippage_bps: float | None = None,
        model_name: str | None = None,
        task: str | None = None,  # "classify" | "regress"
    ) -> None:
        trade_interval_code = trade_interval_code or getattr(
            config, "trade_interval", "5m"
        )
        start_str = start_str or getattr(config, "start_str", "60 days ago UTC")
        self.best_metric = best_metric or getattr(config, "best_metric", "sharpe_like")
        self.sweep_cfg = sweep_cfg or SweepConfig(
            *tuple(
                map(
                    float,
                    (getattr(config, "threshold_sweep", "0.50:0.90:0.005").split(":")),
                )
            )
        )
        self.calib_window_bars = int(
            getattr(config, "calib_window_bars", calib_window_bars)
        )
        self.fees_bps = float(
            getattr(config, "fees_bps", fees_bps if fees_bps is not None else 10.0)
        )
        self.slippage_bps = float(
            getattr(
                config,
                "slippage_bps",
                slippage_bps if slippage_bps is not None else 5.0,
            )
        )
        self.pred_ret_scale_bps = float(
            getattr(config, "pred_ret_scale_bps", 20.0)
        )  # for regression → pseudo-prob

        self.task = (task or getattr(config, "model_task", "classify")).lower()
        if self.task not in {"classify", "regress"}:
            raise ValueError("task must be 'classify' or 'regress'")

        self.client = Client(
            getattr(config, "api_key", None),
            getattr(config, "api_secret", None),
            testnet=getattr(config, "is_test_net", False),
        )
        self.symbol = symbol
        self.interval_code = trade_interval_code
        self.interval = _map_trade_interval_to_binance(trade_interval_code)
        self.interval_ms = INTERVAL_TO_MS[self.interval]
        self.retrain_every = max(1, retrain_every)

        # HistoryManager already defaults to include FNG/On-chain
        self.data = HistoryManager(
            client=self.client,
            symbol=self.symbol,
            interval=self.interval,
            start_str=start_str,
            timelag=timelag,
        )

        # Model selection
        raw_model_name = model_name or getattr(config, "model_name", "hgb")

        # If regression task is requested, map common classifier names → regressors
        if self.task == "regress":
            reg_name_map = {
                "hgb": "hgb_reg",
                "rf": "rf_reg",
                "logreg": "linreg",
                "sgdlog": "linreg",
                "linsvc": "svr",
                "voting_soft": "hgb_reg",
                "stacking": "hgb_reg",
                "metalabel": "hgb_reg",
                # allow passing native reg names too
                "hgb_reg": "hgb_reg",
                "rf_reg": "rf_reg",
                "linreg": "linreg",
                "svr": "svr",
                "bilstm": "bilstm",
                "gru_lstm": "gru_lstm",
                "hybrid_transformer": "hybrid_transformer",
            }
            self.model_name = reg_name_map.get(raw_model_name, "hgb_reg")
        else:
            self.model_name = raw_model_name

        # Sequence models available for both tasks
        self.seq_models = {"bilstm", "gru_lstm", "hybrid_transformer"}
        self.use_sequence = self.model_name in self.seq_models

        self.window = max(2, timelag)
        self.position = PositionManager(
            client=self.client, symbol=self.symbol, sizing=SizingConfig()
        )
        self._bars_since_retrain = 0

        # thresholds
        self._current_threshold = 0.50  # for classification
        self._current_ret_threshold_bps = 0.0  # for regression

        # model holder
        self.model: ModelManager | None = None

    # ---------- Calibration ----------

    def _calibrate_threshold(self) -> None:
        cost_bps = 2.0 * (self.fees_bps + self.slippage_bps)

        if self.task == "classify":
            # Build probability series p_all and forward return window r (fraction)
            X, _ = self.data.dataset(target="direction")
            feat_cols = list(X.columns)

            fwd_ret_series = (
                self.data.df_ohlcv["close"]
                .pct_change()
                .shift(-1)
                .reindex(self.data.df_features.index)
            )

            if self.use_sequence:
                X_seq = _make_rolling_windows(
                    self.data.df_features, feat_cols, self.window
                )
                if len(X_seq) == 0:
                    return
                p_all = self.model.pipeline.predict_proba(X_seq)[:, 1]  # type: ignore

                idx_last = np.arange(self.window - 1, self.window - 1 + len(X_seq))
                r = fwd_ret_series.iloc[idx_last].to_numpy(dtype=float, copy=False)
            else:
                p_all = self.model.pipeline.predict_proba(X)[:, 1]  # type: ignore
                r = fwd_ret_series.to_numpy(dtype=float, copy=False)

            # align / mask NaNs
            m = min(len(r), len(p_all))
            if m <= 0:
                return
            r = r[-m:]
            p_all = p_all[-m:]
            mask = ~np.isnan(r)
            if not mask.any():
                return
            p_win = p_all[mask][-self.calib_window_bars :]
            r_win = r[mask][-self.calib_window_bars :]

            best = choose_best_threshold_for_window(
                p_up_window=p_win,
                fwd_ret_window=r_win,
                interval_code=self.interval_code,
                fees_bps=self.fees_bps,
                slippage_bps=self.slippage_bps,
                sweep=self.sweep_cfg,
                best_metric=self.best_metric,
            )
            self._current_threshold = float(best.get("threshold", 0.50))
            print(
                f"[CALIBRATION] clf thr={self._current_threshold:.3f} "
                f"metric={self.best_metric} value={best.get(self.best_metric)} trades={best.get('trades')}"
            )

        else:  # regress
            # Use predicted next-bar return (bps) and true next-bar return (bps)
            X_ret, y_ret_bps = self.data.dataset(target="return_bps")
            if len(X_ret) == 0:
                return

            if self.use_sequence:
                feat_cols = list(X_ret.columns)
                X_seq = _make_rolling_windows(
                    self.data.df_features, feat_cols, self.window
                )
                if len(X_seq) == 0:
                    return
                yhat_bps_all = self.model.pipeline.predict(X_seq)  # type: ignore
                y_seq = np.asarray(y_ret_bps)[self.window - 1 :]
                y_seq = y_seq[-len(X_seq) :]
            else:
                yhat_bps_all = self.model.pipeline.predict(X_ret)  # type: ignore
                y_seq = np.asarray(y_ret_bps)

            # use last calib_window_bars
            yhat_win = np.asarray(yhat_bps_all)[-self.calib_window_bars :]
            ytrue_win = np.asarray(y_seq)[-self.calib_window_bars :]

            best = _sweep_on_predicted_return(
                yhat_bps=yhat_win,
                fwd_ret_bps=ytrue_win,
                cost_bps=cost_bps,
                best_metric=self.best_metric,
            )
            self._current_ret_threshold_bps = float(best.get("threshold", 0.0))
            print(
                f"[CALIBRATION] reg thr={self._current_ret_threshold_bps:.1f}bps "
                f"(+ cost {cost_bps:.1f}bps) metric={self.best_metric} "
                f"value={best.get(self.best_metric)} trades={best.get('trades')}"
            )

    # ---------- Bootstrap / Retrain ----------

    def bootstrap(self) -> None:
        print("Loading historical data...")
        self.data.load_history()

        if self.task == "classify":
            X, y = self.data.dataset(target="direction")
            self.model = ModelManager(
                predictor_cols=self.data.predictor_cols,
                model_name=self.model_name,
                input_kind="sequence" if self.use_sequence else "tabular",
                task="classify",
            )
            print(
                f"Dataset ready: {len(X)} rows, {len(self.data.predictor_cols)} features (task=classify)"
            )
            if self.use_sequence:
                feat_cols = list(X.columns)
                X_seq = _make_rolling_windows(
                    self.data.df_features, feat_cols, self.window
                )
                if len(X_seq) == 0:
                    raise RuntimeError(f"Not enough data for window={self.window}")
                y_seq = np.asarray(y)[self.window - 1 :]
                y_seq = y_seq[-len(X_seq) :]
                acc = self.model.train(X_seq, y_seq)
            else:
                acc = self.model.train(X, y)
            print(f"Initial test accuracy: {acc:.3f}")

        else:  # regress
            X, y_bps = self.data.dataset(target="return_bps")
            self.model = ModelManager(
                predictor_cols=self.data.predictor_cols,
                model_name=self.model_name,
                input_kind="sequence" if self.use_sequence else "tabular",
                task="regress",
            )
            print(
                f"Dataset ready: {len(X)} rows, {len(self.data.predictor_cols)} features (task=regress)"
            )
            if self.use_sequence:
                feat_cols = list(X.columns)
                X_seq = _make_rolling_windows(
                    self.data.df_features, feat_cols, self.window
                )
                if len(X_seq) == 0:
                    raise RuntimeError(f"Not enough data for window={self.window}")
                y_seq = np.asarray(y_bps)[self.window - 1 :]
                y_seq = y_seq[-len(X_seq) :]
                r2 = self.model.train_regression(X_seq, y_seq)
            else:
                r2 = self.model.train_regression(X, y_bps)
            print(f"Initial test R^2: {r2:.3f}")

        self._calibrate_threshold()

    def _update_and_maybe_retrain(self) -> None:
        self.data.update_with_latest(limit=3)
        self._bars_since_retrain += 1
        if self._bars_since_retrain >= self.retrain_every:
            try:
                if self.task == "classify":
                    X, y = self.data.dataset(target="direction")
                    if self.use_sequence:
                        feat_cols = list(X.columns)
                        X_seq = _make_rolling_windows(
                            self.data.df_features, feat_cols, self.window
                        )
                        if len(X_seq) == 0:
                            raise RuntimeError(
                                f"Not enough data for window={self.window}"
                            )
                        y_seq = np.asarray(y)[self.window - 1 :]
                        y_seq = y_seq[-len(X_seq) :]
                        acc = self.model.train(X_seq, y_seq)
                    else:
                        acc = self.model.train(X, y)
                    print(f"Retrained (clf). Test accuracy: {acc:.3f}")
                else:
                    X, y_bps = self.data.dataset(target="return_bps")
                    if self.use_sequence:
                        feat_cols = list(X.columns)
                        X_seq = _make_rolling_windows(
                            self.data.df_features, feat_cols, self.window
                        )
                        if len(X_seq) == 0:
                            raise RuntimeError(
                                f"Not enough data for window={self.window}"
                            )
                        y_seq = np.asarray(y_bps)[self.window - 1 :]
                        y_seq = y_seq[-len(X_seq) :]
                        r2 = self.model.train_regression(X_seq, y_seq)
                    else:
                        r2 = self.model.train_regression(X, y_bps)
                    print(f"Retrained (reg). Test R^2: {r2:.3f}")

                self._calibrate_threshold()
            except Exception as e:
                print(f"[WARN] Retrain skipped: {e}")
            self._bars_since_retrain = 0

    # ---------- Run Loop ----------

    def run(self) -> None:
        self.bootstrap()
        next_trade_time = self.data.next_bar_close_time_ms()
        print(f"Next decision at ~{pd.to_datetime(next_trade_time, unit='ms')}")
        while True:
            now_ms = int(time.time() * 1000)
            if now_ms >= next_trade_time:
                while now_ms >= next_trade_time:
                    print(f"\n[BAR CLOSE] {pd.to_datetime(next_trade_time, unit='ms')}")
                    self._update_and_maybe_retrain()
                    self.position.close_long()

                    cost_bps = 2.0 * (self.fees_bps + self.slippage_bps)

                    if self.task == "classify":
                        if self.use_sequence:
                            X, _ = self.data.dataset(target="direction")
                            feat_cols = list(X.columns)
                            try:
                                X_win = _last_window(
                                    self.data.df_features, feat_cols, self.window
                                )
                                last_price = float(self.data.df_ohlcv["close"].iloc[-1])
                                p_up = self.model.predict_proba_up(X_win)
                            except Exception as e:
                                print(f"[WARN] Prediction skipped: {e}")
                                p_up = 0.5
                                last_price = float(self.data.df_ohlcv["close"].iloc[-1])
                        else:
                            X_new = self.data.latest_features_row()
                            last_price = float(X_new["CLOSE"].iloc[0])
                            try:
                                p_up = self.model.predict_proba_up(X_new)
                            except Exception as e:
                                print(f"[WARN] Prediction skipped: {e}")
                                p_up = 0.5

                        p_down = 1.0 - p_up
                        print(
                            f"Predicted up/down: {p_up:.3f}/{p_down:.3f}  thr={self._current_threshold:.3f}"
                        )
                        if p_up >= self._current_threshold:
                            qty = self.position.compute_quantity_kelly(p_up, last_price)
                            print(f"Computed qty (kelly-capped): {qty}")
                            self.position.open_long(qty)
                        else:
                            print("Signal below threshold; staying flat.")

                    else:  # regress
                        last_price = float(self.data.df_ohlcv["close"].iloc[-1])
                        try:
                            if self.use_sequence:
                                X_ret, _ = self.data.dataset(target="return_bps")
                                feat_cols = list(X_ret.columns)
                                X_win = _last_window(
                                    self.data.df_features, feat_cols, self.window
                                )
                                pred_bps = float(
                                    self.model.pipeline.predict(X_win)[0]
                                )  # type: ignore
                            else:
                                X_new = self.data.latest_features_row()
                                pred_bps = float(
                                    self.model.pipeline.predict(X_new)[0]
                                )  # type: ignore
                        except Exception as e:
                            print(f"[WARN] Prediction skipped: {e}")
                            pred_bps = 0.0

                        trigger = self._current_ret_threshold_bps + cost_bps
                        print(
                            f"Predicted next-bar return: {pred_bps:.2f} bps | "
                            f"trigger >= {self._current_ret_threshold_bps:.1f} + cost {cost_bps:.1f} = {trigger:.1f} bps"
                        )
                        if pred_bps >= trigger:
                            # turn predicted bps into a pseudo-prob for Kelly sizing
                            p_proxy = _logistic_from_bps(
                                pred_bps, scale_bps=self.pred_ret_scale_bps
                            )
                            qty = self.position.compute_quantity_kelly(
                                p_proxy, last_price
                            )
                            print(
                                f"Computed qty from expected return (kelly via p≈{p_proxy:.3f}): {qty}"
                            )
                            self.position.open_long(qty)
                        else:
                            print("Expected return below threshold; staying flat.")

                    next_trade_time += self.interval_ms

                # (Optional) show balances
                try:
                    usdt_bal = self.client.get_asset_balance(asset="USDT")
                    base_asset = self.symbol.replace("USDT", "")
                    base_bal = self.client.get_asset_balance(asset=base_asset)
                    print(
                        f"Balances: USDT={usdt_bal['free']} {base_asset}={base_bal['free']}"
                    )
                except Exception as e:
                    print(f"[WARN] Balance fetch failed: {e}")

            try:
                time.sleep(1)
            except Exception:
                pass


if __name__ == "__main__":
    bot = TradeBot(
        symbol=getattr(config, "symbol", "BTCUSDT"),
        trade_interval_code=getattr(config, "trade_interval", "1h"),
        start_str=getattr(config, "start_str", "720 days ago UTC"),
        timelag=getattr(config, "timelag", 16),
        retrain_every=getattr(config, "retrain_every", 30),
        model_name=getattr(config, "model_name", "hgb"),
        task=getattr(config, "model_task", "classify"),
    )
    bot.run()
