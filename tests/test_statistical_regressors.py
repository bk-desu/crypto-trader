import numpy as np
import pandas as pd
import pytest

from managers.model_manager import ModelManager


@pytest.fixture
def sample_regression_frame():
    rng = np.random.default_rng(42)
    n = 240
    f1 = rng.normal(scale=1.2, size=n)
    f2 = 0.5 * f1 + rng.normal(scale=0.8, size=n)
    noise = rng.normal(scale=0.3, size=n)
    lagged_f2 = np.concatenate([[f2[0]], f2[:-1]])
    y = 0.6 * f1 + 0.3 * lagged_f2 + noise
    df = pd.DataFrame({"f1": f1, "f2": f2})
    return df, y


@pytest.mark.parametrize("model_name", ["sarimax", "var", "markov_switching"])
def test_statsmodels_regressors_smoke(sample_regression_frame, model_name):
    df, y = sample_regression_frame
    mm = ModelManager(
        predictor_cols=list(df.columns),
        model_name=model_name,
        input_kind="tabular",
        task="regress",
        nn_epochs=1,
        nn_batch_size=8,
    )
    pipe = mm._build_pipeline_reg()
    X_train, X_test = df.iloc[:-30], df.iloc[-30:]
    y_train = y[:-30]
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert preds.shape == (30,)
    assert np.all(np.isfinite(preds))


def test_garch_regressor_smoke(sample_regression_frame):
    pytest.importorskip("arch")
    df, y = sample_regression_frame
    mm = ModelManager(
        predictor_cols=list(df.columns),
        model_name="garch",
        input_kind="tabular",
        task="regress",
        nn_epochs=1,
        nn_batch_size=8,
    )
    pipe = mm._build_pipeline_reg()
    X_train, X_test = df.iloc[:-30], df.iloc[-30:]
    y_train = y[:-30]
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert preds.shape == (30,)
    assert np.all(np.isfinite(preds))
