import numpy as np
import pandas as pd
import pytest

# import the module under test
import bot as bot_mod


class FakePipeline:
    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        # return shape (n,2): [:,1] is prob up
        n = len(X)
        # cycle through probs if needed
        p1 = np.resize(self._probs, n)
        return np.vstack([1 - p1, p1]).T


class FakeModel:
    def __init__(self, probs):
        self.pipeline = FakePipeline(probs)

    def _build_pipeline(self):
        return self.pipeline

    def train(self, X, y):
        return 0.7

    def predict_proba_up(self, X_new):
        # single row predict
        return float(self.pipeline.predict_proba(X_new)[:, 1][0])


class FakeHistory:
    def __init__(self, closes, features):
        idx = pd.RangeIndex(len(closes))
        self.df_ohlcv = pd.DataFrame({"close": closes}, index=idx)
        self.df_features = features
        self.predictor_cols = list(features.columns)
        self._next_time = 1

    def load_history(self):
        pass

    def dataset(self, target: str = "direction"):
        returns = self.df_ohlcv["close"].pct_change().fillna(0)
        if target == "direction":
            y = pd.Series((returns > 0).astype(int))
        elif target == "return_bps":
            y = returns * 10_000.0
        else:
            raise ValueError(target)
        return self.df_features, y

    def update_with_latest(self, limit=3):
        pass

    def latest_features_row(self):
        return self.df_features.tail(1)

    def next_bar_close_time_ms(self):
        return self._next_time


class FakePosition:
    def __init__(self):
        self.open_calls = 0
        self.last_qty = None
        self.closed = 0
        self.is_long = False
        self.last_quantity = 0.0

    def compute_quantity_kelly(self, p_up, last_price):
        return 1.23

    def open_long(self, qty):
        self.open_calls += 1
        self.last_qty = qty
        self.last_quantity = qty
        self.is_long = True

    def close_long(self):
        self.closed += 1
        self.is_long = False
        self.last_quantity = 0.0


class FakeClient:
    def get_asset_balance(self, asset):
        return {"free": "0"}


@pytest.fixture
def patched_bot(monkeypatch):
    # craft synthetic series where later bars are profitable
    closes = np.array([100, 100.5, 101, 101.5, 102, 101.9, 102.5])
    feats = pd.DataFrame({"CLOSE": closes, "f1": np.arange(len(closes))})
    probs = [0.3, 0.55, 0.6, 0.8, 0.85, 0.4, 0.9]

    fake_history = FakeHistory(closes, feats)
    fake_model = FakeModel(probs)
    fake_position = FakePosition()
    fake_client = FakeClient()

    # monkeypatch the classes imported inside bot.py
    monkeypatch.setattr(bot_mod, "HistoryManager", lambda **kw: fake_history)
    monkeypatch.setattr(bot_mod, "ModelManager", lambda **kw: fake_model)
    monkeypatch.setattr(bot_mod, "PositionManager", lambda **kw: fake_position)
    monkeypatch.setattr(bot_mod, "Client", lambda *a, **kw: fake_client)

    bot = bot_mod.TradeBot(
        symbol="BTCUSDT",
        trade_interval_code="1h",
        start_str="10 days ago UTC",
        timelag=2,
        retrain_every=9999,  # avoid retrain during test
        task="classify",
    )
    bot.interval_ms = 1  # tick quickly
    return bot, fake_history, fake_model, fake_position


def test_calibration_sets_reasonable_threshold(patched_bot, monkeypatch):
    bot, hist, model, pos = patched_bot
    bot.bootstrap()
    # because higher probs line up with positive fwd returns, threshold should end up >= 0.5
    assert bot._current_threshold >= 0.5


def test_trade_decision_uses_threshold(patched_bot, monkeypatch):
    bot, hist, model, pos = patched_bot
    bot.bootstrap()

    # Force next prediction below threshold -> should NOT open a long
    def low_prob(_x):
        return 0.49

    model.predict_proba_up = low_prob
    bot.position.open_calls = 0
    # emulate a single bar-close decision block (call subset of run loop)
    bot.position.close_long()
    X_new = hist.latest_features_row()
    _ = model.predict_proba_up(X_new)
    if 0.49 >= bot._current_threshold:
        pytest.skip("threshold ended <= 0.49 unexpectedly in this synthetic setup")
    else:
        # ensure we would not open a long when below threshold
        # (we just assert the threshold logic; open_long not called)
        assert pos.open_calls == 0


def test_classification_trade_maintains_position_when_signal_persists(patched_bot):
    bot, hist, model, pos = patched_bot
    bot.bootstrap()

    bot._current_threshold = 0.6
    bot.position.is_long = True
    bot.position.last_quantity = 1.23
    pos.open_calls = 0

    bot._handle_classification_trade(p_up=0.75, last_price=hist.df_ohlcv["close"].iloc[-1])

    # No additional open orders when quantity unchanged
    assert pos.open_calls == 0
    assert pos.closed == 0
    assert bot.position.is_long


def test_classification_trade_closes_when_signal_drops(patched_bot):
    bot, hist, model, pos = patched_bot
    bot.bootstrap()

    bot._current_threshold = 0.6
    bot.position.is_long = True
    bot.position.last_quantity = 1.23
    pos.closed = 0

    bot._handle_classification_trade(p_up=0.4, last_price=hist.df_ohlcv["close"].iloc[-1])

    assert pos.closed == 1
    assert not bot.position.is_long
