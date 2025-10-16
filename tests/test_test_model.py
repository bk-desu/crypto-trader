import numpy as np
import pandas as pd

import backtest as tm


def test_parse_start_list_mixed():
    res = tm.parse_start_list("365d, 2021-01-01, 90d, 90 days ago UTC")
    assert "365 days ago UTC" in res
    assert "2021-01-01" in res
    assert "90 days ago UTC" in res
    assert res.count("90 days ago UTC") == 2


def test_time_split_indices_shape():
    tr, te = tm.time_split_indices(100, 0.2)
    assert len(tr) == 80 and len(te) == 20
    assert tr[-1] == 79 and te[0] == 80


def test_parse_sweep_to_config():
    cfg = tm.parse_sweep("0.5:0.8:0.1")
    assert cfg.start == 0.5 and cfg.stop == 0.8 and cfg.step == 0.1


def test_evaluate_combo_smoke(monkeypatch):
    # Build minimal fakes to avoid network / sklearn complexities
    class FakeClient:
        pass

    # Synthetic data
    closes = np.array([100, 101, 102, 103, 104, 103.5, 104.5, 104.7], dtype=float)
    idx = pd.RangeIndex(len(closes))
    df_ohlcv = pd.DataFrame({"close": closes}, index=idx)
    df_features = pd.DataFrame(
        {
            "CLOSE": closes,
            "f1": np.linspace(0, 1, len(closes)),
        },
        index=idx,
    )

    # Fake HistoryManager
    class FakeHistory:
        def __init__(self, **kw):
            self.df_ohlcv = df_ohlcv.copy()
            self.df_features = df_features.copy()
            self.predictor_cols = list(self.df_features.columns)

        def load_history(self):
            pass

        def dataset(self, target="direction"):
            if target == "return_bps":
                ret = self.df_ohlcv["close"].pct_change().fillna(0) * 10_000.0
                return self.df_features, ret
            y = pd.Series(
                (self.df_ohlcv["close"].pct_change().fillna(0) > 0).astype(int)
            )
            return self.df_features, y

    # Fake ModelManager pipeline with deterministic probabilities
    class FakePipeline:
        def predict_proba(self, X):
            # prob up increases with f1
            p1 = X["f1"].to_numpy()
            return np.vstack([1 - p1, p1]).T

        def fit(self, X, y):
            pass

    class FakeRegPipeline:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.full(len(X), 0.001)

    class FakeModel:
        def __init__(self, **kw):
            self.pipeline = FakePipeline()
            self.reg_pipeline = FakeRegPipeline()

        def _build_pipeline(self):
            return self.pipeline

        def _build_pipeline_reg(self):
            return self.reg_pipeline

        def train(self, X, y):
            return 0.66

    # Patch external deps inside module
    monkeypatch.setattr(tm, "HistoryManager", FakeHistory)
    monkeypatch.setattr(tm, "ModelManager", FakeModel)
    # map_interval returns anything non-None from dict—just bypass
    monkeypatch.setattr(tm, "map_interval", lambda code: code)

    res = tm.evaluate_combo(
        symbol="BTCUSDT",
        start_str="60d",
        interval_code="1h",
        timelag=10,
        model_name="logreg",
        class_weight=None,
        split_mode="time",
        test_size=0.25,
        out_dir=".",
        save_datasets=False,
        save_predictions=False,
        sweep_cfg=tm.SweepConfig(0.5, 0.9, 0.1),
        fees_bps=10.0,
        slippage_bps=5.0,
        label_mode="direction",
        ret_bps=0.0,
        client=FakeClient(),
        best_metric="sharpe_like",
        task="regress",
    )

    # sanity checks
    for k in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "best_threshold",
        "trades",
        "total_net_return",
        "sharpe_like",
        "avg_net_ret_per_bar",
    ]:
        assert k in res
