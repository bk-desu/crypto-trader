from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)  # NEW (ARIMA)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

# Deep learning (sequence models) via SciKeras + Keras
try:
    from scikeras.wrappers import KerasClassifier, KerasRegressor  # type: ignore
except Exception:  # pragma: no cover
    KerasClassifier = None  # type: ignore
    KerasRegressor = None  # type: ignore

# Prefer the standalone Keras package (used by transformers/tf-keras) but
# gracefully fall back to tensorflow.keras if it's not available.
keras = None  # type: ignore
layers = None  # type: ignore
try:  # pragma: no cover - executed in most DL-enabled environments
    import keras as _keras_mod  # type: ignore
    from keras import layers as _keras_layers  # type: ignore

    keras = _keras_mod  # type: ignore
    layers = _keras_layers  # type: ignore
    try:  # ensure TF backend when using Keras 3
        if hasattr(keras, "config"):
            backend = keras.config.backend()  # type: ignore[attr-defined]
            if backend != "tensorflow":  # pragma: no branch
                keras.config.set_backend("tensorflow")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback silently if API differs
        pass
except Exception:  # pragma: no cover
    try:
        from tensorflow import keras as _keras_mod  # type: ignore
        from tensorflow.keras import layers as _keras_layers  # type: ignore

        keras = _keras_mod  # type: ignore
        layers = _keras_layers  # type: ignore
    except Exception:  # pragma: no cover
        keras = None  # type: ignore
        layers = None  # type: ignore

try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
except Exception:  # pragma: no cover
    SARIMAX = None  # type: ignore
try:
    from statsmodels.tsa.api import VAR  # type: ignore
except Exception:  # pragma: no cover
    VAR = None  # type: ignore
try:
    from statsmodels.tsa.regime_switching.markov_regression import (  # type: ignore
        MarkovRegression,
    )
except Exception:  # pragma: no cover
    MarkovRegression = None  # type: ignore

# ---- model/type names (keep your existing classifier names) ----
ClassifierName = Literal[
    "logreg",
    "sgdlog",
    "rf",
    "hgb",
    "linsvc",
    "bilstm",
    "gru_lstm",
    "hybrid_transformer",
    "voting_soft",
    "stacking",
    "metalabel",
    "arima",
]
# Simple, practical regressors to start
RegressorName = Literal[
    "hgb_reg",
    "rf_reg",
    "linreg",
    "svr",
    "arima",
    "bilstm",
    "gru_lstm",
    "hybrid_transformer",
    "sarimax",
    "var",
    "garch",
    "markov_switching",
]

ModelName = Union[ClassifierName, RegressorName]
Task = Literal["classify", "regress"]

tf.get_logger().setLevel("ERROR")


def _needs_scaling(name: str) -> bool:
    """Scale for linear / margin models; trees don't need it."""
    return name in {
        # classifiers
        "logreg",
        "sgdlog",
        "linsvc",
        "voting_soft",
        "stacking",
        # regressors
        "linreg",
        "svr",
    }


class GARCHVolatilityTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self, returns_col: str, out_col: str = "garch_vol", p: int = 1, q: int = 1
    ):
        self.returns_col = returns_col
        self.out_col = out_col
        self.p = p
        self.q = q

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        Xc = X.copy()
        try:
            from arch import arch_model
        except Exception as e:
            raise ImportError(
                "Install 'arch' to use GARCHVolatilityTransformer: pip install arch"
            ) from e

        s = pd.Series(Xc[self.returns_col].astype(float))
        keep = s.dropna()
        if keep.empty:
            Xc[self.out_col] = np.nan
            return Xc

        am = arch_model(keep.values, vol="GARCH", p=self.p, q=self.q, rescale=False)
        res = am.fit(disp="off")
        vol = pd.Series(res.conditional_volatility, index=keep.index)
        aligned = vol.reindex(s.index).ffill().bfill()
        Xc[self.out_col] = aligned.values
        return Xc


class SequenceMaker(TransformerMixin, BaseEstimator):
    def __init__(self, fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return X
        if isinstance(X, pd.DataFrame) and self.fn is not None:
            arr = self.fn(X)
            if not (isinstance(arr, np.ndarray) and arr.ndim == 3):
                raise ValueError(
                    "sequence_maker must return a 3D numpy array (n, t, d)."
                )
            return arr
        raise RuntimeError(
            "For sequence models, provide X as a 3D numpy array or pass sequence_maker=callable."
        )


def _as_2d_array(X: Union[pd.DataFrame, np.ndarray, None]) -> Optional[np.ndarray]:
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float, copy=False)
        return np.asarray(arr, dtype=float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 3:
        n, t, d = arr.shape
        return arr.reshape(n, t * d)
    return arr


class MetaLabelingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, base: BaseEstimator, meta: BaseEstimator, threshold: float = 0.5
    ):
        self.base = base
        self.meta = meta
        self.threshold = threshold

    def fit(self, X, y):
        self.base.fit(X, y)
        p = self.base.predict_proba(X)[:, 1]
        mask = p >= self.threshold
        if isinstance(X, pd.DataFrame):
            X_mask = X.loc[mask]
        else:
            X_mask = X[mask]
        y_mask = (
            y[mask] if isinstance(y, (pd.Series, np.ndarray)) else np.array(y)[mask]
        )
        if len(y_mask) == 0:
            self.meta.fit(X, y)
        else:
            self.meta.fit(X_mask, y_mask)
        return self

    def predict_proba(self, X):
        pb = self.base.predict_proba(X)[:, 1]
        meta_p = np.full_like(pb, 0.5, dtype=float)
        mask = pb >= self.threshold
        if mask.any():
            Xm = X.loc[mask] if isinstance(X, pd.DataFrame) else X[mask]
            meta_p[mask] = self.meta.predict_proba(Xm)[:, 1]
        final = pb * meta_p
        return np.vstack([1 - final, final]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _ensure_keras():
    if keras is None or layers is None:
        raise ImportError(
            "Sequence models require TensorFlow + SciKeras. Install:\n  pip install tensorflow scikeras"
        )


def _bilstm_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.Bidirectional(layers.LSTM(64))(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _gru_lstm_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _hybrid_transformer_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.LayerNormalization()(inp)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=max(8, d // 2))(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ffn = keras.Sequential([layers.Dense(128, activation="relu"), layers.Dense(d)])
    x = layers.Add()([x, ffn(x)])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _bilstm_reg_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.Bidirectional(layers.LSTM(64))(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _gru_lstm_reg_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _hybrid_transformer_reg_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.LayerNormalization()(inp)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=max(8, d // 2))(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ffn = keras.Sequential([layers.Dense(128, activation="relu"), layers.Dense(d)])
    x = layers.Add()([x, ffn(x)])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class ARIMAClassifier(BaseEstimator, ClassifierMixin):  # sklearn-compatible
    """
    ARIMA over a binary target (0/1). Ignores X and models y_t directly.
    Probabilities are obtained by applying a logistic link to the ARIMA forecast.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 0)):
        self.order = order
        self.model_ = None
        self.results_ = None
        self.class_prior_ = None  # fallback prior

    def fit(self, X, y):
        if ARIMA is None:
            raise ImportError(
                "Install statsmodels to use ARIMA: pip install statsmodels"
            )
        y_arr = np.asarray(y, dtype=float).ravel()
        # store class prior for safety
        self.class_prior_ = float(np.mean(y_arr)) if y_arr.size else 0.5
        # fit ARIMA on the binary series (as a numeric time series)
        self.model_ = ARIMA(y_arr, order=self.order)
        self.results_ = self.model_.fit()
        return self

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        if self.results_ is None:
            raise RuntimeError("ARIMAClassifier is not fitted.")
        n = len(X) if hasattr(X, "__len__") else 1
        # Forecast n steps ahead; map to (0,1) via logistic; fallback to prior if needed
        fc = np.asarray(self.results_.forecast(steps=int(n)), dtype=float).ravel()
        p1 = self._sigmoid(fc)  # logistic link to keep in [0,1]
        # guard against NaNs
        if not np.isfinite(p1).all():
            p1 = np.full(
                int(n), self.class_prior_ if self.class_prior_ is not None else 0.5
            )
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ARIMARegressor(BaseEstimator, RegressorMixin):
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 0)):
        self.order = order
        self.model_ = None
        self.results_ = None

    def fit(self, X, y):
        if ARIMA is None:
            raise ImportError(
                "Install statsmodels to use ARIMA: pip install statsmodels"
            )
        y_arr = np.asarray(y, dtype=float).ravel()
        self.model_ = ARIMA(y_arr, order=self.order)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.results_ is None:
            raise RuntimeError("ARIMA model is not fitted.")
        n_steps = len(X) if hasattr(X, "__len__") else 1
        # One-step-ahead style: forecast n_steps into the future
        fc = self.results_.forecast(steps=int(n_steps))
        return np.asarray(fc, dtype=float).ravel()


class SARIMAXRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = "c",
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model_ = None
        self.results_ = None
        self._exog_dim: Optional[int] = None

    def fit(self, X, y):
        if SARIMAX is None:
            raise ImportError(
                "Install statsmodels to use SARIMAXRegressor: pip install statsmodels"
            )
        y_arr = np.asarray(y, dtype=float).ravel()
        exog = _as_2d_array(X)
        if exog is not None and len(exog) != len(y_arr):
            raise ValueError("X and y must have the same number of rows for SARIMAX.")
        self.model_ = SARIMAX(
            y_arr,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.results_ = self.model_.fit(disp=False)
        self._exog_dim = None if exog is None else exog.shape[1]
        return self

    def predict(self, X):
        if self.results_ is None:
            raise RuntimeError("SARIMAX model is not fitted.")
        exog = _as_2d_array(X)
        if self._exog_dim is None:
            exog = None
        elif exog is None:
            raise ValueError("Exogenous features were used in training and are required for prediction.")
        elif exog.shape[1] != self._exog_dim:
            raise ValueError(
                f"Expected {self._exog_dim} exogenous features at predict time, got {exog.shape[1]}"
            )
        steps = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        forecast = self.results_.get_forecast(steps=steps, exog=exog)
        return np.asarray(forecast.predicted_mean, dtype=float)


class VARRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        maxlags: Optional[int] = None,
        ic: Optional[str] = None,
        trend: str = "c",
    ):
        self.maxlags = maxlags
        self.ic = ic
        self.trend = trend
        self.model_ = None
        self.results_ = None
        self._endog_history: Optional[np.ndarray] = None
        self._k_ar: int = 1

    def fit(self, X, y):
        if VAR is None:
            raise ImportError("Install statsmodels to use VARRegressor: pip install statsmodels")
        y_arr = np.asarray(y, dtype=float).ravel()
        exog = _as_2d_array(X)
        if exog is None:
            endog = y_arr.reshape(-1, 1)
        else:
            if len(exog) != len(y_arr):
                raise ValueError("X and y must have the same number of rows for VAR.")
            endog = np.column_stack([y_arr.reshape(-1, 1), exog])
        self.model_ = VAR(endog)
        self.results_ = self.model_.fit(maxlags=self.maxlags, ic=self.ic, trend=self.trend)
        self._k_ar = int(self.results_.k_ar)
        if self._k_ar <= 0:
            self._k_ar = 1
        self._endog_history = endog[-self._k_ar :]
        return self

    def predict(self, X):
        if self.results_ is None or self._endog_history is None:
            raise RuntimeError("VAR model is not fitted.")
        steps = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        forecasts = self.results_.forecast(self._endog_history, steps=steps)
        return np.asarray(forecasts[:, 0], dtype=float)


class GARCHRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "AR",
        lags: int | List[int] | None = 1,
        vol: str = "GARCH",
    ):
        self.p = p
        self.q = q
        self.mean = mean
        self.lags = lags
        self.vol = vol
        self.model_ = None
        self.results_ = None

    def fit(self, X, y):
        try:
            from arch import arch_model  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install 'arch' to use GARCHRegressor: pip install arch") from exc

        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.size < max(10, (self.p + self.q + 1)):
            raise ValueError("Not enough observations to fit GARCH model.")
        self.model_ = arch_model(
            y_arr,
            mean=self.mean,
            lags=self.lags,
            vol=self.vol,
            p=self.p,
            q=self.q,
            rescale=False,
        )
        self.results_ = self.model_.fit(disp="off", update_freq=0)
        return self

    def predict(self, X):
        if self.results_ is None:
            raise RuntimeError("GARCH model is not fitted.")
        steps = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        fc = self.results_.forecast(horizon=steps, reindex=False)
        mean_df = fc.mean
        preds = mean_df.iloc[-1].to_numpy(dtype=float)
        return np.asarray(preds, dtype=float)


class MarkovSwitchingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        k_regimes: int = 2,
        trend: str = "c",
        switching_variance: bool = True,
        switching_trend: bool = False,
        markov_order: int = 1,
        maxiter: int = 200,
        tol: float = 1e-6,
    ):
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_variance = switching_variance
        self.switching_trend = switching_trend
        self.markov_order = markov_order
        self.maxiter = maxiter
        self.tol = tol
        self.model_ = None
        self.results_ = None

    def fit(self, X, y):
        if MarkovRegression is None:
            raise ImportError(
                "Install statsmodels to use MarkovSwitchingRegressor: pip install statsmodels"
            )
        y_arr = np.asarray(y, dtype=float).ravel()
        self.model_ = MarkovRegression(
            y_arr,
            k_regimes=self.k_regimes,
            trend=self.trend,
            switching_variance=self.switching_variance,
            switching_trend=self.switching_trend,
            order=self.markov_order,
        )
        self.results_ = self.model_.fit(disp=False, maxiter=self.maxiter, tol=self.tol)
        try:
            self._last_pred_mean = float(self.results_.predict()[-1])
        except Exception:
            self._last_pred_mean = float(np.mean(y_arr))
        return self

    def predict(self, X):
        if self.results_ is None:
            raise RuntimeError("Markov-switching model is not fitted.")
        steps = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        if steps <= 0:
            return np.empty(0, dtype=float)
        try:
            fc = self.results_.forecast(steps=steps)
            return np.asarray(fc, dtype=float)
        except NotImplementedError:
            return np.full(steps, self._last_pred_mean, dtype=float)


class ModelManager:
    """
    Flexible sklearn pipeline manager with multiple back-ends.
    - task='classify' keeps your current behavior (probability of UP).
    - task='regress' predicts next-bar return (e.g., in bps).
    """

    def __init__(
        self,
        predictor_cols: List[str],
        model_name: ModelName = "logreg",
        class_weight: Optional[str] = None,  # "balanced" or None
        calibrate_svc: bool = True,
        random_state: int = 1,
        input_kind: Literal["tabular", "sequence"] = "tabular",
        sequence_maker: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        nn_epochs: int = 20,
        nn_batch_size: int = 32,
        metalabel_base: ClassifierName = "hgb",
        metalabel_meta: ClassifierName = "logreg",
        metalabel_threshold: float = 0.5,
        garch_returns_col: Optional[str] = None,
        garch_out_col: str = "garch_vol",
        garch_pq: Tuple[int, int] = (1, 1),
        voting_members: Optional[List[ClassifierName]] = None,
        stacking_members: Optional[List[ClassifierName]] = None,
        # NEW
        task: Task = "classify",
    ) -> None:
        self.predictor_cols = predictor_cols
        self.numeric_cols = predictor_cols[:-2]  # original convention
        self.pattern_cols = predictor_cols[-2:]  # PATT_3OUT, PATT_CMB
        self.model_name: str = model_name  # accept both classifier & regressor names
        self.class_weight = class_weight
        self.calibrate_svc = calibrate_svc
        self.random_state = random_state

        self.input_kind = input_kind
        self.sequence_maker = sequence_maker
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size

        self.metalabel_base = metalabel_base
        self.metalabel_meta = metalabel_meta
        self.metalabel_threshold = metalabel_threshold

        self.garch_returns_col = garch_returns_col
        self.garch_out_col = garch_out_col
        self.garch_pq = garch_pq

        self.voting_members = voting_members or ["logreg", "rf", "hgb"]
        self.stacking_members = stacking_members or ["logreg", "rf", "hgb"]

        self.task: Task = task
        self.pipeline: Optional[Pipeline] = None
        self.last_test_accuracy: Optional[float] = None  # accuracy (clf) or R^2 (reg)

    # -------------------- classifier builders --------------------
    def _simple_estimator(self, name: ClassifierName):
        if name == "logreg":
            return LogisticRegression(
                max_iter=10000,
                solver="saga",
                C=0.25,
                class_weight=self.class_weight,
                random_state=self.random_state,
                tol=1e-3,
            )
        if name == "sgdlog":
            base = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return CalibratedClassifierCV(base, cv=3)
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=None,
                max_bins=255,
                random_state=self.random_state,
            )
        if name == "linsvc":
            svc = LinearSVC(
                class_weight=self.class_weight, random_state=self.random_state
            )
            return CalibratedClassifierCV(svc, cv=3)
        if name == "bilstm":
            _ensure_keras()
            if KerasClassifier is None:
                raise ImportError(
                    "Sequence classifiers require SciKeras. Install: pip install scikeras"
                )
            return KerasClassifier(
                model=_bilstm_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "gru_lstm":
            _ensure_keras()
            if KerasClassifier is None:
                raise ImportError(
                    "Sequence classifiers require SciKeras. Install: pip install scikeras"
                )
            return KerasClassifier(
                model=_gru_lstm_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "hybrid_transformer":
            _ensure_keras()
            if KerasClassifier is None:
                raise ImportError(
                    "Sequence classifiers require SciKeras. Install: pip install scikeras"
                )
            return KerasClassifier(
                model=_hybrid_transformer_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "voting_soft":
            members = [(m, self._simple_estimator(m)) for m in self.voting_members]
            return VotingClassifier(estimators=members, voting="soft")
        if name == "stacking":
            members = [(m, self._simple_estimator(m)) for m in self.stacking_members]
            final = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return StackingClassifier(
                estimators=members, final_estimator=final, passthrough=False
            )
        if name == "metalabel":
            base = self._simple_estimator(self.metalabel_base)
            meta = self._simple_estimator(self.metalabel_meta)
            return MetaLabelingClassifier(
                base=base, meta=meta, threshold=self.metalabel_threshold
            )
        if name == "arima":
            return ARIMAClassifier()
        raise ValueError(f"Unknown estimator name: {name}")

    def _build_pipeline_clf(self) -> Pipeline:
        steps: List[Tuple[str, object]] = []

        if self.input_kind == "tabular" and self.garch_returns_col is not None:
            steps.append(
                (
                    "garch",
                    GARCHVolatilityTransformer(
                        returns_col=self.garch_returns_col,
                        out_col=self.garch_out_col,
                        p=self.garch_pq[0],
                        q=self.garch_pq[1],
                    ),
                )
            )
            if self.garch_out_col not in self.numeric_cols:
                self.numeric_cols = list(self.numeric_cols) + [self.garch_out_col]

        if self.input_kind == "sequence":
            steps.append(("to_seq", SequenceMaker(self.sequence_maker)))
            steps.append(("clf", self._simple_estimator(self.model_name)))  # type: ignore[arg-type]
            return Pipeline(steps)

        prep = (
            ColumnTransformer(
                transformers=[("scale", StandardScaler(), self.numeric_cols)],
                remainder="passthrough",
            )
            if _needs_scaling(self.model_name)
            else "passthrough"
        )

        steps.append(("prep", prep))
        steps.append(("clf", self._simple_estimator(self.model_name)))  # type: ignore[arg-type]
        return Pipeline(steps)

    # -------------------- regressor builders --------------------
    def _simple_estimator_reg(self, name: RegressorName):
        if name == "hgb_reg":
            return HistGradientBoostingRegressor(random_state=self.random_state)
        if name == "rf_reg":
            return RandomForestRegressor(
                n_estimators=400, n_jobs=-1, random_state=self.random_state
            )
        if name == "linreg":
            return Ridge(alpha=1.0, random_state=self.random_state)
        if name == "svr":
            return SVR(kernel="rbf")  # scaled upstream
        if name == "bilstm":
            _ensure_keras()
            if KerasRegressor is None:
                raise ImportError(
                    "Sequence regressors require SciKeras. Install: pip install scikeras"
                )
            return KerasRegressor(
                model=_bilstm_reg_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "gru_lstm":
            _ensure_keras()
            if KerasRegressor is None:
                raise ImportError(
                    "Sequence regressors require SciKeras. Install: pip install scikeras"
                )
            return KerasRegressor(
                model=_gru_lstm_reg_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "hybrid_transformer":
            _ensure_keras()
            if KerasRegressor is None:
                raise ImportError(
                    "Sequence regressors require SciKeras. Install: pip install scikeras"
                )
            return KerasRegressor(
                model=_hybrid_transformer_reg_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "sarimax":
            return SARIMAXRegressor()
        if name == "var":
            return VARRegressor()
        if name == "garch":
            return GARCHRegressor()
        if name == "markov_switching":
            return MarkovSwitchingRegressor()
        if name == "arima":
            return ARIMARegressor()
        raise ValueError(f"Unknown regressor name: {name}")

    def _build_pipeline_reg(self) -> Pipeline:
        steps: List[Tuple[str, object]] = []

        if self.input_kind == "tabular" and self.garch_returns_col is not None:
            steps.append(
                (
                    "garch",
                    GARCHVolatilityTransformer(
                        returns_col=self.garch_returns_col,
                        out_col=self.garch_out_col,
                        p=self.garch_pq[0],
                        q=self.garch_pq[1],
                    ),
                )
            )
            if self.garch_out_col not in self.numeric_cols:
                self.numeric_cols = list(self.numeric_cols) + [self.garch_out_col]

        if self.input_kind == "sequence":
            steps.append(("to_seq", SequenceMaker(self.sequence_maker)))
            steps.append(("reg", self._simple_estimator_reg(self.model_name)))  # type: ignore[arg-type]
            return Pipeline(steps)

        prep = (
            ColumnTransformer(
                transformers=[("scale", StandardScaler(), self.numeric_cols)],
                remainder="passthrough",
            )
            if _needs_scaling(self.model_name)
            else "passthrough"
        )

        steps.append(("prep", prep))
        steps.append(("reg", self._simple_estimator_reg(self.model_name)))  # type: ignore[arg-type]
        return Pipeline(steps)

    # -------------------- training / inference --------------------
    def train(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Classification training (backward compatible).
        Returns test accuracy.
        """
        if self.task != "classify":
            raise RuntimeError("Use train_regression() when task='regress'.")

        if pd.Series(y).nunique() < 2:
            raise RuntimeError("Target has fewer than 2 classes; wait for more data.")

        stratify = y if isinstance(y, (pd.Series, np.ndarray)) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=stratify
        )
        self.pipeline = self._build_pipeline_clf()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.last_test_accuracy = float(acc)
        return self.last_test_accuracy

    def predict_proba_up(self, X_one: Union[pd.DataFrame, np.ndarray]) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        proba = self.pipeline.predict_proba(X_one)[0][1]
        return float(proba)

    # --- regression ---
    def train_regression(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Regression training for next-bar return (e.g., bps).
        Returns R^2 on a held-out split (and stores it in last_test_accuracy).
        """
        if self.task != "regress":
            raise RuntimeError("Use train() when task='classify'.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.pipeline = self._build_pipeline_reg()
        self.pipeline.fit(X_train, y_train)
        y_hat = self.pipeline.predict(X_test)
        r2 = r2_score(y_test, y_hat)
        self.last_test_accuracy = float(r2)
        return self.last_test_accuracy

    def predict_return(self, X_one: Union[pd.DataFrame, np.ndarray]) -> float:
        """
        Predict next-bar return (same unit you trained on; recommended: bps).
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        return float(self.pipeline.predict(X_one)[0])
