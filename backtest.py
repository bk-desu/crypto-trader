import argparse
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from binance.client import Client
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from evaluations.evaluator import choose_best_threshold_for_window
from evaluations.metrics import SweepConfig, safe_mape_pct
from managers.history_manager import HistoryManager
from managers.model_manager import ModelManager
from metrics.deflated_sharpe import deflated_sharpe_ratio

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def map_interval(code: str) -> str:
    from binance.client import Client

    m = {
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
    if code not in m:
        raise ValueError(f"Unsupported interval code: {code}")
    return m[code]


def make_windows_from_df(
    df: pd.DataFrame, feat_cols: list[str], window: int, stride: int = 1
) -> np.ndarray:
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    if len(arr) < window:
        return np.empty((0, window, arr.shape[1]), dtype=np.float32)
    sw = sliding_window_view(arr, window_shape=window, axis=0)
    return sw[::stride]


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for sub in ["metrics", "datasets", "predictions"]:
        os.makedirs(os.path.join(path, sub), exist_ok=True)


def time_split_indices(n_rows: int, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    test_len = max(1, int(n_rows * test_frac))
    train_len = max(1, n_rows - test_len)
    idx = np.arange(n_rows)
    return idx[:train_len], idx[train_len:]


def parse_sweep(arg: str) -> SweepConfig:
    a, b, c = map(float, arg.split(":"))
    if not (0.0 <= a < b <= 1.0) or c <= 0:
        raise ValueError(
            "threshold sweep must be 'start:stop:step' with 0<=start<stop<=1 and step>0"
        )
    return SweepConfig(start=a, stop=b, step=c)


def parse_start_list(arg: str) -> List[str]:
    out = []
    for token in [s.strip() for s in arg.split(",") if s.strip()]:
        if token.lower().endswith("d") and token[:-1].isdigit():
            out.append(f"{int(token[:-1])} days ago UTC")
        else:
            out.append(token)
    return out


def sweep_on_predicted_return(
    yhat_bps: np.ndarray,
    fwd_ret_bps: np.ndarray,
    cost_bps: float,
    best_metric: str,
    thr_grid_bps: np.ndarray | None = None,
) -> Dict:
    """
    Long-only: trade when predicted return >= (threshold + cost).
    Metrics are net of costs (bps).
    """
    if thr_grid_bps is None:
        thr_grid_bps = np.arange(0.0, 60.1, 1.0)  # 0..60 bps

    trial_srs: List[float] = []

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
        if std > 0 and np.isfinite(sharpe_like):
            trial_srs.append(sharpe_like)

        # Sortino-like: mean / downside std (only negative net bars)
        downside = net[net < 0.0]
        downside_std = float(np.nanstd(downside)) if downside.size > 0 else np.nan
        sortino_like = (
            float(avg / (downside_std + 1e-9))
            if not np.isnan(downside_std)
            else float("nan")
        )

        metric = {
            "total_net_return": total,
            "avg_net_ret_per_bar": avg,
            "sharpe_like": sharpe_like,
            "sortino_like": sortino_like,
        }.get(best_metric, sharpe_like)

        if metric > best.get(best_metric, float("-inf")):
            best = {
                "threshold": float(thr),
                "trades": trades,
                "total_net_return": total,
                "avg_net_ret_per_bar": avg,
                "sharpe_like": sharpe_like,
                "sortino_like": sortino_like,
            }
    if best["total_net_return"] == float("-inf"):
        best.update(
            {
                "threshold": 0.0,
                "trades": 0,
                "total_net_return": 0.0,
                "avg_net_ret_per_bar": float("nan"),
                "sharpe_like": float("nan"),
                "sortino_like": float("nan"),
            }
        )
    best["trial_srs"] = trial_srs
    return best


def _build_price_frame(
    data,
    indices: np.ndarray,
    split_label: Literal["train", "test"],
    y_true_bps: np.ndarray,
    y_pred_bps: np.ndarray,
    *,
    symbol: str,
    model: str,
    task: str,
    interval_code: str,
    start_str: str,
    predicted_prob: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compose a tidy frame with actual/predicted prices for plotting."""

    if len(indices) == 0:
        return pd.DataFrame()

    idx_arr = np.asarray(indices)
    closes = (
        data.df_ohlcv["close"].reindex(idx_arr).to_numpy(dtype=float, copy=False)
    )
    feature_time_ms = (
        data.df_ohlcv["open_time"].reindex(idx_arr).to_numpy(dtype=float, copy=False)
    )
    target_time_ms = (
        data.df_ohlcv["open_time"].shift(-1).reindex(idx_arr).to_numpy(
            dtype=float, copy=False
        )
    )
    actual_next_close = (
        data.df_ohlcv["close"].shift(-1).reindex(idx_arr).to_numpy(
            dtype=float, copy=False
        )
    )

    y_true = np.asarray(y_true_bps, dtype=float)
    y_pred = np.asarray(y_pred_bps, dtype=float)

    mask = ~(
        np.isnan(closes)
        | np.isnan(feature_time_ms)
        | np.isnan(target_time_ms)
        | np.isnan(actual_next_close)
        | np.isnan(y_true)
        | np.isnan(y_pred)
    )

    if predicted_prob is not None:
        prob_arr = np.asarray(predicted_prob, dtype=float)
        mask &= ~np.isnan(prob_arr)
    else:
        prob_arr = None

    if not mask.any():
        return pd.DataFrame()

    idx_arr = idx_arr[mask]
    closes = closes[mask]
    feature_time_ms = feature_time_ms[mask].astype(np.int64, copy=False)
    target_time_ms = target_time_ms[mask].astype(np.int64, copy=False)
    actual_next_close = actual_next_close[mask]
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if prob_arr is not None:
        prob_arr = prob_arr[mask]

    predicted_next_close = closes * (1.0 + (y_pred / 10_000.0))

    df = pd.DataFrame(
        {
            "symbol": symbol,
            "model": model,
            "task": task,
            "interval": interval_code,
            "start_str": start_str,
            "split": split_label,
            "sample_index": idx_arr.astype(int, copy=False),
            "feature_time_ms": feature_time_ms,
            "target_time_ms": target_time_ms,
            "current_close": closes,
            "actual_next_close": actual_next_close,
            "predicted_next_close": predicted_next_close,
            "actual_return_bps": y_true,
            "predicted_return_bps": y_pred,
        }
    )

    df["feature_time_iso"] = (
        pd.to_datetime(df["feature_time_ms"], unit="ms", utc=True)
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    df["target_time_iso"] = (
        pd.to_datetime(df["target_time_ms"], unit="ms", utc=True)
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    if prob_arr is not None:
        df["predicted_prob_up"] = prob_arr
    else:
        df["predicted_prob_up"] = np.nan

    return df


def evaluate_combo(
    symbol: str,
    start_str: str,
    interval_code: str,
    timelag: int,
    model_name: str,
    class_weight: Optional[str],
    split_mode: str,
    test_size: float,
    out_dir: str,
    save_datasets: bool,
    save_predictions: bool,
    sweep_cfg: SweepConfig,
    fees_bps: float,
    slippage_bps: float,
    label_mode: str,
    ret_bps: float,
    best_metric: str,
    client: Client,
    task: str,
) -> Dict:
    b_interval = map_interval(interval_code)
    print(
        f"\n--- Interval {interval_code} / start {start_str} / model {model_name} / task {task} ---"
    )

    data = HistoryManager(
        client=client,
        symbol=symbol,
        interval=b_interval,
        start_str=start_str,
        timelag=timelag,
        # HistoryManager defaults include FNG/on-chain already enabled
    )
    data.load_history()

    # ================= REGRESSION PATH =================
    if task == "regress":
        # Predict next-bar return in bps
        X, y_ret = data.dataset(target="return_bps")
        original_index = X.index.to_numpy()

        # map common classifier names to sensible regressors if provided
        reg_name_map = {
            "hgb": "hgb_reg",
            "rf": "rf_reg",
            "logreg": "linreg",
            "sgdlog": "linreg",
            "linsvc": "svr",
            "voting_soft": "hgb_reg",
            "stacking": "hgb_reg",
            "metalabel": "hgb_reg",
            "hgb_reg": "hgb_reg",
            "rf_reg": "rf_reg",
            "linreg": "linreg",
            "svr": "svr",
            "arima": "arima",
            "bilstm": "bilstm",
            "gru_lstm": "gru_lstm",
            "hybrid_transformer": "hybrid_transformer",
            "sarimax": "sarimax",
            "var": "var",
            "garch": "garch",
            "markov_switching": "markov_switching",
        }
        reg_model_name = reg_name_map.get(model_name, "hgb_reg")

        seq_models = {"bilstm", "gru_lstm", "hybrid_transformer"}
        use_sequence = reg_model_name in seq_models

        if use_sequence:
            window = max(2, timelag)
            stride = 1
            feat_cols = list(X.columns)
            X_seq = make_windows_from_df(
                data.df_features, feat_cols, window=window, stride=stride
            )
            if len(X_seq) == 0:
                raise RuntimeError(
                    f"Not enough rows ({len(X)}) to form any {window}-length windows."
                )
            idx_last = (window - 1) + np.arange(len(X_seq)) * stride
            y_aligned = y_ret.to_numpy(dtype=float, copy=False)[idx_last]
            aligned_index = original_index[idx_last]
            y_aligned_series = pd.Series(y_aligned, index=aligned_index)
            X_nd = X_seq
        else:
            y_aligned = y_ret.to_numpy(dtype=float, copy=False)
            aligned_index = original_index
            y_aligned_series = pd.Series(y_aligned, index=aligned_index)
            X_nd = X.reset_index(drop=True)

        model = ModelManager(
            predictor_cols=list(X.columns),
            model_name=reg_model_name,
            input_kind="sequence" if use_sequence else "tabular",
            task="regress",
        )

        # time split and fit ONLY on train
        n = len(X_nd)
        idx_train, idx_test = time_split_indices(n, test_size)
        if use_sequence:
            X_train = X_nd[idx_train]  # type: ignore[index]
            X_test = X_nd[idx_test]  # type: ignore[index]
            y_train_series = y_aligned_series.iloc[idx_train]
            y_test_series = y_aligned_series.iloc[idx_test]
        else:
            X_train = X_nd.iloc[idx_train]  # type: ignore[assignment]
            X_test = X_nd.iloc[idx_test]  # type: ignore[assignment]
            y_train_series = y_aligned_series.iloc[idx_train]
            y_test_series = y_aligned_series.iloc[idx_test]

        y_train = y_train_series.to_numpy(dtype=float, copy=False)
        y_test = y_test_series.to_numpy(dtype=float, copy=False)

        model.pipeline = model._build_pipeline_reg()
        model.pipeline.fit(X_train, y_train)
        yhat_train = np.asarray(model.pipeline.predict(X_train), dtype=float)
        yhat_test = np.asarray(model.pipeline.predict(X_test), dtype=float)

        from sklearn.metrics import r2_score

        r2 = float(r2_score(y_test, yhat_test))

        rmse_bps = float(np.sqrt(mean_squared_error(y_test, yhat_test)))

        cost_bps = 2.0 * (fees_bps + slippage_bps)
        best = sweep_on_predicted_return(
            yhat_bps=np.asarray(yhat_test, dtype=float),
            fwd_ret_bps=np.asarray(y_test, dtype=float),
            cost_bps=cost_bps,
            best_metric=best_metric,
        )
        # for deflated sharpe ratio
        thr = float(best.get("threshold", 0.0))
        take_best = yhat_test >= (thr + cost_bps)
        net_bps_best = (np.asarray(y_test, dtype=float)[take_best]) - cost_bps
        dsr = deflated_sharpe_ratio(net_bps_best, best.get("trial_srs", []))

        mape_pct = safe_mape_pct(y_test, yhat_test)

        # artifacts
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{symbol}_{interval_code}_{reg_model_name}_{ts_tag}"

        aligned_index_arr = y_aligned_series.index.to_numpy()
        train_indices = aligned_index_arr[idx_train]
        test_indices = aligned_index_arr[idx_test]

        price_frames = [
            _build_price_frame(
                data,
                train_indices,
                "train",
                y_train,
                yhat_train,
                symbol=symbol,
                model=reg_model_name,
                task="regress",
                interval_code=interval_code,
                start_str=start_str,
            ),
            _build_price_frame(
                data,
                test_indices,
                "test",
                y_test,
                yhat_test,
                symbol=symbol,
                model=reg_model_name,
                task="regress",
                interval_code=interval_code,
                start_str=start_str,
            ),
        ]

        price_frames = [df for df in price_frames if not df.empty]
        price_rel_path = ""
        if price_frames:
            price_df = (
                pd.concat(price_frames, ignore_index=True)
                .sort_values("target_time_ms")
                .reset_index(drop=True)
            )
            os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
            price_rel_path = os.path.join(
                "predictions", f"PRICE_{start_str.replace(' ', '_')}_{tag}.csv"
            )
            price_df.to_csv(os.path.join(out_dir, price_rel_path), index=False)
        if save_datasets:
            os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
            data.df_ohlcv.to_csv(
                os.path.join(
                    out_dir,
                    "datasets",
                    f"OHLCV_{start_str.replace(' ', '_')}_{tag}.csv",
                ),
                index=False,
            )
            data.df_features.to_csv(
                os.path.join(
                    out_dir,
                    "datasets",
                    f"FEATURES_{start_str.replace(' ', '_')}_{tag}.csv",
                ),
                index=True,
            )
        if save_predictions:
            os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
            pred_df = pd.DataFrame(
                {"p_up": np.nan, "y_true": np.nan, "fwd_ret": np.nan}
            )
            pred_df.loc[: len(yhat_test) - 1, "fwd_ret"] = y_test
            pred_df.loc[: len(yhat_test) - 1, "p_up"] = yhat_test
            pred_df.to_csv(
                os.path.join(
                    out_dir,
                    "predictions",
                    f"PRED_{start_str.replace(' ', '_')}_{tag}.csv",
                ),
                index=False,
            )

        return {
            "symbol": symbol,
            "interval": interval_code,
            "start_str": start_str,
            "timelag": timelag,
            "model": reg_model_name,
            "rows": n,
            "split_mode": split_mode,
            "test_size": test_size,
            "class_weight": "",
            "label_mode": "return_bps",
            "ret_bps": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "confusion_matrix": [],
            "best_threshold": float(best.get("threshold", 0.0)),
            "trades": int(best.get("trades", 0)),
            "hit_rate": float("nan"),
            "avg_net_ret_per_bar": float(best.get("avg_net_ret_per_bar", 0.0)),
            "avg_net_ret_per_trade": float("nan"),
            "total_net_return": float(best.get("total_net_return", 0.0)),
            "sharpe_like": float(best.get("sharpe_like", float("nan"))),
            "deflated_sharpe": float(dsr),
            "sortino_like": float(best.get("sortino_like", float("nan"))),
            "cost_roundtrip": float(2.0 * ((fees_bps + slippage_bps) / 10_000.0)),
            "r2": float(r2),
            "mae_bps": float(mean_absolute_error(y_test, yhat_test)),
            "rmse_bps": float(rmse_bps),
            "mape_pct": float(mape_pct),
            "price_track_path": price_rel_path,
        }

    # ================= CLASSIFICATION PATH (original) =================
    X, y_dir = data.dataset(target="direction")
    original_index_cls = X.index.to_numpy()

    # forward returns aligned with features (for money metrics)
    fwd_ret = (
        data.df_ohlcv["close"]
        .pct_change()
        .shift(-1)
        .reindex(data.df_features.index)
        .values
    )

    if label_mode == "ret_gt_bps":
        thr = ret_bps / 10_000.0
        y = (fwd_ret > thr).astype(int)
        mask_ok = ~np.isnan(fwd_ret)
        X, y, fwd_ret = X[mask_ok], y[mask_ok], fwd_ret[mask_ok]
    else:
        y = y_dir.values

    seq_models = {"bilstm", "gru_lstm", "hybrid_transformer"}
    use_sequence = model_name in seq_models

    if use_sequence:
        window = max(2, timelag)
        stride = 1
        feat_cols = list(X.columns)
        X_seq = make_windows_from_df(
            data.df_features, feat_cols, window=window, stride=stride
        )
        n_seq = len(X_seq)
        if n_seq == 0:
            raise RuntimeError(
                f"Not enough rows ({len(X)}) to form any {window}-length windows."
            )

        idx_last = (window - 1) + np.arange(n_seq) * stride
        y = np.asarray(y)[idx_last]
        fwd_ret = fwd_ret[idx_last]
        aligned_index_cls = original_index_cls[idx_last]
        X_nd = X_seq
    else:
        aligned_index_cls = original_index_cls
        X_nd = X

    n = len(X_nd)
    if n < 200:
        print(f"[WARN] Only {n} rows; results may be noisy.")

    aligned_index_arr = np.asarray(aligned_index_cls)
    fwd_ret_arr = np.asarray(fwd_ret, dtype=float)
    y_arr = np.asarray(y)

    model = ModelManager(
        predictor_cols=list(X.columns),
        model_name=model_name,
        class_weight=class_weight,
        random_state=1,
        input_kind="sequence" if use_sequence else "tabular",
        sequence_maker=None,
        task="classify",
    )

    price_frames: List[pd.DataFrame] = []

    if split_mode == "random":
        test_acc = model.train(X_nd, pd.Series(y))
        pipe = model.pipeline
        p_up_all = pipe.predict_proba(X_nd)[:, 1]
        y_pred_all = (p_up_all >= 0.5).astype(int)

        acc = test_acc
        prec = precision_score(y, y_pred_all, zero_division=0)
        rec = recall_score(y, y_pred_all, zero_division=0)
        f1 = f1_score(y, y_pred_all, zero_division=0)
        try:
            auc = roc_auc_score(y, p_up_all)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y, y_pred_all).tolist()

        test_mask = np.ones(n, dtype=bool)
        p_up_test, y_test, fwd_test = p_up_all, y, fwd_ret_arr

        pos_mask = y_arr == 1
        neg_mask = y_arr == 0
        pos_mean = np.nanmean(fwd_ret_arr[pos_mask]) if pos_mask.any() else np.nan
        neg_mean = np.nanmean(fwd_ret_arr[neg_mask]) if neg_mask.any() else np.nan
        if np.isnan(pos_mean):
            pos_mean = np.nanmean(fwd_ret_arr)
        if np.isnan(neg_mean):
            neg_mean = np.nanmean(fwd_ret_arr)
        if np.isnan(pos_mean):
            pos_mean = 0.0
        if np.isnan(neg_mean):
            neg_mean = 0.0
        expected_ret_all = p_up_all * pos_mean + (1.0 - p_up_all) * neg_mean
        price_frames.append(
            _build_price_frame(
                data,
                aligned_index_arr,
                "test",
                fwd_ret_arr * 10_000.0,
                expected_ret_all * 10_000.0,
                symbol=symbol,
                model=model_name,
                task="classify",
                interval_code=interval_code,
                start_str=start_str,
                predicted_prob=p_up_all,
            )
        )

    else:  # time split
        idx_train, idx_test = time_split_indices(n, test_size)
        X_train_split = X_nd[idx_train] if use_sequence else X.iloc[idx_train]
        X_test_split = X_nd[idx_test] if use_sequence else X.iloc[idx_test]
        y_train_split = y[idx_train]
        y_test = y[idx_test]

        model.pipeline = model._build_pipeline_clf()
        model.pipeline.fit(X_train_split, y_train_split)

        p_up_train = model.pipeline.predict_proba(X_train_split)[:, 1]
        p_up_test = model.pipeline.predict_proba(X_test_split)[:, 1]
        y_pred = (p_up_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, p_up_test)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_mask = np.zeros(n, dtype=bool)
        test_mask[idx_test] = True
        fwd_train = fwd_ret_arr[idx_train]
        fwd_test = fwd_ret_arr[idx_test]

        pos_mask = y_arr[idx_train] == 1
        neg_mask = y_arr[idx_train] == 0
        pos_mean = np.nanmean(fwd_train[pos_mask]) if pos_mask.any() else np.nan
        neg_mean = np.nanmean(fwd_train[neg_mask]) if neg_mask.any() else np.nan
        if np.isnan(pos_mean):
            pos_mean = np.nanmean(fwd_train)
        if np.isnan(neg_mean):
            neg_mean = np.nanmean(fwd_train)
        if np.isnan(pos_mean):
            pos_mean = 0.0
        if np.isnan(neg_mean):
            neg_mean = 0.0

        expected_ret_train = p_up_train * pos_mean + (1.0 - p_up_train) * neg_mean
        expected_ret_test = p_up_test * pos_mean + (1.0 - p_up_test) * neg_mean
        price_frames.extend(
            [
                _build_price_frame(
                    data,
                    aligned_index_arr[idx_train],
                    "train",
                    fwd_train * 10_000.0,
                    expected_ret_train * 10_000.0,
                    symbol=symbol,
                    model=model_name,
                    task="classify",
                    interval_code=interval_code,
                    start_str=start_str,
                    predicted_prob=p_up_train,
                ),
                _build_price_frame(
                    data,
                    aligned_index_arr[idx_test],
                    "test",
                    fwd_test * 10_000.0,
                    expected_ret_test * 10_000.0,
                    symbol=symbol,
                    model=model_name,
                    task="classify",
                    interval_code=interval_code,
                    start_str=start_str,
                    predicted_prob=p_up_test,
                ),
            ]
        )

    # threshold sweep (probability) w/ costs
    best = choose_best_threshold_for_window(
        p_up_window=p_up_test,
        fwd_ret_window=fwd_test,
        interval_code=interval_code,
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
        sweep=sweep_cfg,
        best_metric=best_metric,
    )

    # log loss (guarded)
    try:
        # p_up_test and y_test are already defined in both split modes
        logloss = float(log_loss(y_test, p_up_test, labels=[0, 1]))
    except Exception:
        logloss = float("nan")

    # sortino-like: rebuild per-trade net series at chosen threshold
    try:
        thr = float(best.get("threshold", 0.50))
        take = p_up_test >= thr
        trades = int(np.sum(take))
        if trades > 0:
            # fwd_test is fraction return; convert to bps, then subtract round-trip costs
            cost_bps = 2.0 * (fees_bps + slippage_bps)
            net_bps = (fwd_test[take] * 10_000.0) - cost_bps
            avg_net = float(np.nanmean(net_bps))
            downside = net_bps[net_bps < 0.0]
            downside_std = float(np.nanstd(downside)) if downside.size > 0 else np.nan
            sortino_like = (
                float(avg_net / (downside_std + 1e-9))
                if not np.isnan(downside_std)
                else float("nan")
            )
        else:
            sortino_like = float("nan")
    except Exception:
        sortino_like = float("nan")

    # artifacts
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{symbol}_{interval_code}_{model_name}_{ts_tag}"

    price_frames = [df for df in price_frames if not df.empty]
    price_rel_path = ""
    if price_frames:
        price_df = (
            pd.concat(price_frames, ignore_index=True)
            .sort_values("target_time_ms")
            .reset_index(drop=True)
        )
        os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
        price_rel_path = os.path.join(
            "predictions", f"PRICE_{start_str.replace(' ', '_')}_{tag}.csv"
        )
        price_df.to_csv(os.path.join(out_dir, price_rel_path), index=False)

    if save_datasets:
        os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
        HistoryManager_df_ohlcv = data.df_ohlcv.copy()
        HistoryManager_df_features = data.df_features.copy()
        HistoryManager_df_ohlcv.to_csv(
            os.path.join(
                out_dir, "datasets", f"OHLCV_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=False,
        )
        HistoryManager_df_features.to_csv(
            os.path.join(
                out_dir, "datasets", f"FEATURES_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=True,
        )

    if save_predictions:
        os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
        pred_df = pd.DataFrame(
            {
                "is_test": np.where(test_mask, 1, 0),
                "p_up": np.nan,
                "y_true": np.nan,
                "fwd_ret": np.nan,
            }
        )
        if split_mode == "random":
            pred_df.loc[:, "p_up"] = p_up_test
            pred_df.loc[:, "y_true"] = y_test
            pred_df.loc[:, "fwd_ret"] = fwd_test
        else:
            pred_df.loc[test_mask, "p_up"] = p_up_test
            pred_df.loc[test_mask, "y_true"] = y_test
            pred_df.loc[test_mask, "fwd_ret"] = fwd_test

        pred_df.to_csv(
            os.path.join(
                out_dir, "predictions", f"PRED_{start_str.replace(' ', '_')}_{tag}.csv"
            ),
            index=False,
        )

    return {
        "symbol": symbol,
        "interval": interval_code,
        "start_str": start_str,
        "timelag": timelag,
        "model": model_name,
        "rows": len(X_nd),
        "split_mode": split_mode,
        "test_size": test_size,
        "class_weight": class_weight or "",
        "label_mode": label_mode,
        "ret_bps": ret_bps if label_mode == "ret_gt_bps" else 0.0,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
        "best_threshold": float(best.get("threshold", 0.50)),
        "trades": int(best.get("trades", 0)),
        "hit_rate": float(best.get("hit_rate", float("nan"))),
        "avg_net_ret_per_bar": float(best.get("avg_net_ret_per_bar", 0.0)),
        "avg_net_ret_per_trade": float(best.get("avg_net_ret_per_trade", float("nan"))),
        "total_net_return": float(best.get("total_net_return", 0.0)),
        "sharpe_like": float(best.get("sharpe_like", float("nan"))),
        "sortino_like": float(sortino_like),
        "cost_roundtrip": float(2.0 * ((fees_bps + slippage_bps) / 10_000.0)),
        "r2": float("nan"),
        "mae_bps": float("nan"),
        "mape_pct": float("nan"),
        "log_loss": float(logloss),
        "price_track_path": price_rel_path,
    }


def main():
    p = argparse.ArgumentParser(
        description="Grid-validate models over intervals * start_strs * models."
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument(
        "--start-list",
        default="365d,720d",
        help="Comma list of windows or absolute dates, e.g. '180d,365d,2021-01-01,90 days ago UTC'.",
    )
    p.add_argument(
        "--intervals", default="1h,4h", help="Comma list, e.g. '5m,15m,1h,4h,1d'."
    )
    p.add_argument(
        "--models",
        default="logreg,rf,hgb,linsvc,bilstm,gru_lstm,hybrid_transformer,voting_soft,stacking,metalabel,arima",
        help="Comma list for classification; for regression you can still pass these and they map to regressors.",
    )
    p.add_argument("--timelag", type=int, default=16)
    p.add_argument("--split-mode", choices=["time", "random"], default="time")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--class-weight", choices=["balanced", "none"], default="none")
    p.add_argument(
        "--label-mode", choices=["direction", "ret_gt_bps"], default="direction"
    )
    p.add_argument(
        "--ret-bps",
        type=float,
        default=0.0,
        help="Return threshold in bps for 'ret_gt_bps' mode.",
    )
    p.add_argument("--out-dir", default="backtest_output")
    p.add_argument("--save-datasets", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument(
        "--threshold-sweep",
        default="0.50:0.90:0.005",
        help="start:stop:step for p_up threshold",
    )
    p.add_argument(
        "--best-metric",
        choices=["sharpe_like", "total_net_return", "avg_net_ret_per_bar"],
        default="sharpe_like",
    )
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=1.5)
    p.add_argument("--task", choices=["classify", "regress"], default="regress")

    args = p.parse_args()

    start_list = parse_start_list(args.start_list)
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sweep_cfg = (
        parse_sweep(args.threshold_sweep) if args.threshold_sweep else SweepConfig()
    )
    class_weight = None if args.class_weight == "none" else "balanced"

    ensure_dirs(args.out_dir)

    client = Client(
        getattr(config, "api_key", None),
        getattr(config, "api_secret", None),
        testnet=False,
    )

    results = []
    for start_str in start_list:
        for interval_code in intervals:
            for model_name in models:
                try:
                    res = evaluate_combo(
                        symbol=args.symbol,
                        start_str=start_str,
                        interval_code=interval_code,
                        timelag=args.timelag,
                        model_name=model_name,
                        class_weight=class_weight,
                        split_mode=args.split_mode,
                        test_size=args.test_size,
                        out_dir=args.out_dir,
                        save_datasets=args.save_datasets,
                        save_predictions=args.save_predictions,
                        sweep_cfg=sweep_cfg,
                        fees_bps=args.fees_bps,
                        slippage_bps=args.slippage_bps,
                        label_mode=args.label_mode,
                        ret_bps=args.ret_bps,
                        best_metric=args.best_metric,
                        client=client,
                        task=args.task,
                    )
                    results.append(res)
                except Exception as e:
                    print(
                        f"[ERROR] start={start_str} interval={interval_code} model={model_name}: {e}"
                    )

    if results:
        df = pd.DataFrame(results)
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.out_dir, "metrics", f"SUMMARY_{args.symbol}_{ts_tag}.csv"
        )
        df.to_csv(out_path, index=False)
        print("\nSummary written to:", out_path)

        # print a compact view that works for both tasks
        cols = [
            "start_str",
            "interval",
            "model",
            "rows",
            "accuracy",
            "r2",
            "log_loss",
            "rmse_bps",
            "mape_pct",
            "avg_net_ret_per_bar",
            "total_net_return",
            "sharpe_like",
            "sortino_like",
            "best_threshold",
            "trades",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        print(df[cols])
    else:
        print("No successful evaluations.")


if __name__ == "__main__":
    main()
