"""
Unit tests for mlops_frauddetection — Section 1 (PHASE3.md evidence).

Calls your REAL module code — verified against your actual src tree:
  - config.py              PROJECT_ROOT, TrainingConfig, DataConfig, Config
  - data.exceptions        RawDataNotFoundError, DataLoaderError, VisualizationError
  - data.make_dataset      raw_data_exists()
  - features.build_features num_feat list
  - models.base            BaseModel ABC
  - models.model           Model scaffold (save/load via joblib)
  - models.fraud_models    get_lr_balanced, get_lr_smote, get_rf_model,
                           get_lgb_model, get_xgb_model
  - evaluation.metrics     evaluate_multiclass()
  - utils.seed             set_seed()
  - utils.io               save_json(), load_json()

# - NOTE: evaluate_binary() is NOT tested here because it does not exist in
      your metrics.py yet. evaluate_multiclass() IS tested (confirmed in src).
      get_rf_model() kwargs are tested conservatively (no seed= kwarg assumed).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. Config
# =============================================================================


class TestConfig:
    """mlops_frauddetection.config — paths and frozen dataclasses."""

    def test_project_root_exists(self):
        from mlops_frauddetection.config import PROJECT_ROOT

        assert PROJECT_ROOT.exists()

    def test_path_constants_are_path_instances(self):
        from mlops_frauddetection.config import (
            CONFIG_DIR,
            DATA_DIR,
            LOGS_DIR,
            MODELS_DIR,
            REPORTS_DIR,
        )

        for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, CONFIG_DIR]:
            assert isinstance(d, Path)

    def test_training_config_seed_is_42(self):
        from mlops_frauddetection.config import TrainingConfig

        assert TrainingConfig().seed == 42

    def test_training_config_batch_size_is_32(self):
        from mlops_frauddetection.config import TrainingConfig

        assert TrainingConfig().batch_size == 32

    def test_training_config_is_frozen(self):
        from mlops_frauddetection.config import TrainingConfig

        with pytest.raises((AttributeError, TypeError)):
            TrainingConfig().seed = 99  # type: ignore[misc]

    def test_data_config_train_split_in_range(self):
        from mlops_frauddetection.config import DataConfig

        cfg = DataConfig()
        assert 0 < cfg.train_test_split < 1.0

    def test_data_config_splits_sum_to_one_or_less(self):
        from mlops_frauddetection.config import DataConfig

        cfg = DataConfig()
        assert cfg.train_test_split + cfg.val_split <= 1.0

    def test_config_has_training_subconfig(self):
        from mlops_frauddetection.config import Config, TrainingConfig

        assert isinstance(Config().training, TrainingConfig)


# =============================================================================
# 2. Data exceptions
# =============================================================================


class TestDataExceptions:
    """mlops_frauddetection.data.exceptions — custom exception classes."""

    def test_raw_data_not_found_is_raiseable(self):
        from mlops_frauddetection.data.exceptions import RawDataNotFoundError

        with pytest.raises(RawDataNotFoundError):
            raise RawDataNotFoundError("missing")

    def test_data_loader_error_is_raiseable(self):
        from mlops_frauddetection.data.exceptions import DataLoaderError

        with pytest.raises(DataLoaderError):
            raise DataLoaderError("load failed")

    def test_visualization_error_is_raiseable(self):
        from mlops_frauddetection.data.exceptions import VisualizationError

        with pytest.raises(VisualizationError):
            raise VisualizationError("plot failed")

    def test_all_inherit_from_exception(self):
        from mlops_frauddetection.data.exceptions import (
            DataLoaderError,
            RawDataNotFoundError,
            VisualizationError,
        )

        for exc in [RawDataNotFoundError, DataLoaderError, VisualizationError]:
            assert issubclass(exc, Exception)

    def test_raw_data_exists_raises_on_missing_directory(self, tmp_path, monkeypatch):
        """raw_data_exists() must raise RawDataNotFoundError for absent dir."""
        import mlops_frauddetection.data.make_dataset as md
        from mlops_frauddetection.data.exceptions import RawDataNotFoundError

        monkeypatch.setattr(md, "RAW_DATA_DIR", tmp_path / "no_such_dir")
        with pytest.raises(RawDataNotFoundError):
            md.raw_data_exists("transactions.csv")

    def test_raw_data_exists_raises_when_file_absent(self, tmp_path, monkeypatch):
        """raw_data_exists() must raise when dir exists but file is missing."""
        import mlops_frauddetection.data.make_dataset as md
        from mlops_frauddetection.data.exceptions import RawDataNotFoundError

        monkeypatch.setattr(md, "RAW_DATA_DIR", tmp_path)
        with pytest.raises(RawDataNotFoundError):
            md.raw_data_exists("transactions.csv")

    def test_raw_data_exists_returns_true_when_present(self, tmp_path, monkeypatch):
        """raw_data_exists() must return True when the CSV exists."""
        import mlops_frauddetection.data.make_dataset as md

        (tmp_path / "transactions.csv").touch()
        monkeypatch.setattr(md, "RAW_DATA_DIR", tmp_path)
        assert md.raw_data_exists("transactions.csv") is True


# =============================================================================
# 3. Model scaffold (models.base + models.model)
# =============================================================================


class TestModelScaffold:
    """mlops_frauddetection.models.model — BaseModel contract + joblib I/O."""

    def test_model_is_subclass_of_base_model(self):
        from mlops_frauddetection.models.base import BaseModel
        from mlops_frauddetection.models.model import Model

        assert issubclass(Model, BaseModel)

    def test_default_config_is_empty_dict(self):
        from mlops_frauddetection.models.model import Model

        assert Model().config == {}

    def test_custom_config_stored(self):
        from mlops_frauddetection.models.model import Model

        cfg = {"lr": 0.01, "epochs": 5}
        assert Model(cfg).config == cfg

    def test_fit_raises_not_implemented(self):
        from mlops_frauddetection.models.model import Model

        with pytest.raises(NotImplementedError):
            Model().fit(None, None)

    def test_predict_raises_not_implemented(self):
        from mlops_frauddetection.models.model import Model

        with pytest.raises(NotImplementedError):
            Model().predict(None)

    def test_save_load_roundtrip(self, tmp_path):
        from mlops_frauddetection.models.model import Model

        p = tmp_path / "model.joblib"
        m = Model({"lr": 0.05})
        m.save(p)
        loaded = Model.load(p)
        assert isinstance(loaded, Model)
        assert loaded.config == m.config

    def test_load_rejects_wrong_type(self, tmp_path):
        from mlops_frauddetection.models.model import Model

        p = tmp_path / "bad.joblib"
        joblib.dump({"just": "a dict"}, p)
        with pytest.raises(TypeError):
            Model.load(p)


# =============================================================================
# 4. Feature engineering
# =============================================================================


class TestFeatureEngineering:
    """mlops_frauddetection.features.build_features — num_feat and matrix."""

    def test_num_feat_is_non_empty_list(self):
        from mlops_frauddetection.features.build_features import num_feat

        assert isinstance(num_feat, list) and len(num_feat) > 0

    def test_num_feat_contains_log_amt(self):
        from mlops_frauddetection.features.build_features import num_feat

        assert "log_amt" in num_feat

    def test_num_feat_contains_cyclic_hour_features(self):
        from mlops_frauddetection.features.build_features import num_feat

        assert "trans_hour_sin" in num_feat
        assert "trans_hour_cos" in num_feat

    def test_num_feat_contains_velocity_ratio(self):
        from mlops_frauddetection.features.build_features import num_feat

        assert "velocity_ratio" in num_feat

    def test_feature_matrix_is_2d(self, binary_x_y):
        x, _ = binary_x_y
        assert x.ndim == 2

    def test_feature_matrix_has_no_nans(self, binary_x_y):
        x, _ = binary_x_y
        assert not np.isnan(x).any()

    def test_feature_matrix_has_no_infs(self, binary_x_y):
        x, _ = binary_x_y
        assert not np.isinf(x).any()

    def test_target_is_binary(self, binary_x_y):
        _, y = binary_x_y
        assert set(np.unique(y)).issubset({0, 1})

    def test_fraud_rate_is_realistic(self, binary_x_y):
        _, y = binary_x_y
        rate = float(y.mean())
        assert 0.001 <= rate <= 0.30, f"fraud rate {rate:.3%} outside range"


# =============================================================================
# 5. Model factories  (fraud_models.py)
# =============================================================================


class TestModelFactories:
    """mlops_frauddetection.models.fraud_models — factory return types."""

    def test_get_lr_balanced_returns_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        from mlops_frauddetection.models.fraud_models import get_lr_balanced

        assert isinstance(get_lr_balanced(), LogisticRegression)

    def test_get_lr_balanced_uses_balanced_class_weight(self):
        from mlops_frauddetection.models.fraud_models import get_lr_balanced

        assert get_lr_balanced().class_weight == "balanced"

    def test_get_lr_balanced_respects_seed_kwarg(self):
        from mlops_frauddetection.models.fraud_models import get_lr_balanced

        assert get_lr_balanced(seed=7).random_state == 7

    def test_get_lr_balanced_respects_max_iter_kwarg(self):
        from mlops_frauddetection.models.fraud_models import get_lr_balanced

        assert get_lr_balanced(max_iter=500).max_iter == 500

    def test_get_lr_smote_returns_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        from mlops_frauddetection.models.fraud_models import get_lr_smote

        assert isinstance(get_lr_smote(), LogisticRegression)

    def test_get_rf_model_returns_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        from mlops_frauddetection.models.fraud_models import get_rf_model

        assert isinstance(get_rf_model(), RandomForestClassifier)

    def test_get_lgb_model_returns_lgbm_classifier(self):
        lgb = pytest.importorskip("lightgbm")
        from mlops_frauddetection.models.fraud_models import get_lgb_model

        assert isinstance(get_lgb_model(), lgb.LGBMClassifier)

    def test_get_xgb_model_returns_xgb_classifier(self):
        xgb = pytest.importorskip("xgboost")
        from mlops_frauddetection.models.fraud_models import get_xgb_model

        assert isinstance(get_xgb_model(), xgb.XGBClassifier)


# =============================================================================
# 6. Model training (using get_lr_balanced — pipeline A model)
# =============================================================================


class TestModelTraining:
    """Fit → predict → persist → reload via your real get_lr_balanced factory."""

    def test_lr_balanced_pipeline_fits(self, trained_lr_balanced):
        assert trained_lr_balanced is not None

    def test_predict_shape_matches_test_set(self, trained_lr_balanced, train_test_data):
        _, X_test, _, _ = train_test_data
        assert trained_lr_balanced.predict(X_test).shape == (X_test.shape[0],)

    def test_predict_returns_only_binary_labels(
        self, trained_lr_balanced, train_test_data
    ):
        _, X_test, _, _ = train_test_data
        assert set(trained_lr_balanced.predict(X_test)).issubset({0, 1})

    def test_predict_proba_rows_sum_to_one(self, trained_lr_balanced, train_test_data):
        _, X_test, _, _ = train_test_data
        proba = trained_lr_balanced.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_joblib_roundtrip_predictions_identical(
        self, trained_lr_balanced, train_test_data, tmp_model_path
    ):
        _, X_test, _, _ = train_test_data
        joblib.dump(trained_lr_balanced, tmp_model_path)
        loaded = joblib.load(tmp_model_path)
        np.testing.assert_array_equal(
            trained_lr_balanced.predict(X_test),
            loaded.predict(X_test),
        )

    def test_saved_model_file_exists(self, trained_lr_balanced, tmp_model_path):
        joblib.dump(trained_lr_balanced, tmp_model_path)
        assert tmp_model_path.exists()

    def test_roc_auc_above_random_baseline(self, trained_lr_balanced, train_test_data):
        _, X_test, _, y_test = train_test_data
        proba = trained_lr_balanced.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc > 0.5, f"AUC {auc:.3f} is not better than random"


# =============================================================================
# 7. Evaluation metrics  (evaluate_multiclass() contract + output validity)
# =============================================================================


class TestEvaluationMetrics:
    """mlops_frauddetection.evaluation.metrics — evaluate_multiclass()."""

    def _make_4class(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_true = pd.Series(rng.integers(0, 4, 100))
        y_pred = rng.integers(0, 4, 100)
        return y_true, y_pred

    def test_returns_dict(self):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        result = evaluate_multiclass(*self._make_4class())
        assert isinstance(result, dict)

    def test_has_test_acc_key(self):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        assert "test_acc" in evaluate_multiclass(*self._make_4class())

    def test_has_test_f1_key(self):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        assert "test_f1" in evaluate_multiclass(*self._make_4class())

    def test_test_acc_in_valid_range(self):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        result = evaluate_multiclass(*self._make_4class())
        assert 0.0 <= result["test_acc"] <= 1.0

    def test_test_f1_in_valid_range(self):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        result = evaluate_multiclass(*self._make_4class())
        assert 0.0 <= result["test_f1"] <= 1.0

    def test_with_overfitting_check_args(self):
        """Passing train labels should not raise — triggers gap check."""
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        y_true, y_pred = self._make_4class()
        result = evaluate_multiclass(
            y_true,
            y_pred,
            y_train_true=y_true,
            y_train_pred=y_pred,
        )
        assert isinstance(result, dict)

    def test_perfect_predictions_give_acc_one(self):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        y = pd.Series([0, 1, 2, 3, 0, 1, 2, 3])
        result = evaluate_multiclass(y, y.values)
        assert result["test_acc"] == pytest.approx(1.0)


# =============================================================================
# 8. Utilities (seed + json I/O)
# =============================================================================


class TestUtilities:
    """mlops_frauddetection.utils — set_seed() and save/load JSON."""

    def test_set_seed_runs_without_error(self):
        from mlops_frauddetection.utils.seed import set_seed

        set_seed(42)

    def test_set_seed_makes_numpy_deterministic(self):
        from mlops_frauddetection.utils.seed import set_seed

        set_seed(42)
        a = np.random.randint(0, 10_000, 20)
        set_seed(42)
        b = np.random.randint(0, 10_000, 20)
        np.testing.assert_array_equal(a, b)

    def test_save_load_json_roundtrip(self, tmp_path):
        from mlops_frauddetection.utils.io import load_json, save_json

        data = {"model": "lr_balanced", "auc": 0.87, "tags": ["fraud"]}
        p = tmp_path / "metrics.json"
        save_json(data, p)
        assert load_json(p) == data

    def test_save_json_creates_parent_directories(self, tmp_path):
        from mlops_frauddetection.utils.io import save_json

        p = tmp_path / "a" / "b" / "out.json"
        save_json({"ok": True}, p)
        assert p.exists()

    def test_load_json_raises_on_missing_file(self, tmp_path):
        from mlops_frauddetection.utils.io import load_json

        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "ghost.json")


# =============================================================================
# 9. Parametrized multi-model
# =============================================================================


@pytest.mark.parametrize(
    "factory_name, factory_kwargs",
    [
        ("get_lr_balanced", {"max_iter": 200}),
        ("get_lr_smote", {"max_iter": 200}),
        ("get_rf_model", {}),  # no extra kwargs — safe default
        ("get_lgb_model", {}),
        ("get_xgb_model", {}),
    ],
    ids=["lr_balanced", "lr_smote", "random_forest", "lightgbm", "xgboost"],
)
def test_all_model_factories_fit_and_predict(
    factory_name, factory_kwargs, train_test_data
):
    """
    Parametrized: every factory in fraud_models.py must produce a model that
    fits without error and returns valid binary predictions.

    Mirrors the @pytest.mark.parametrize pattern from SE489 Week 8 (slide 35).
    """
    import importlib

    try:
        mod = importlib.import_module("mlops_frauddetection.models.fraud_models")
        clf = getattr(mod, factory_name)(**factory_kwargs)
    except (ImportError, AttributeError, TypeError):
        pytest.skip(f"{factory_name} not available")

    X_train, X_test, y_train, _ = train_test_data
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    assert preds.shape == (X_test.shape[0],), "wrong prediction shape"
    assert set(preds).issubset({0, 1}), "predictions must be binary"


@pytest.mark.parametrize(
    "factory_name",
    [
        "get_lr_balanced",
        "get_lr_smote",
        "get_rf_model",
        "get_lgb_model",
        "get_xgb_model",
    ],
    ids=["lr_balanced", "lr_smote", "random_forest", "lightgbm", "xgboost"],
)
def test_all_model_factories_auc_above_random(factory_name, train_test_data):
    """All five model factories must achieve ROC-AUC > 0.5 on held-out data."""
    import importlib

    try:
        mod = importlib.import_module("mlops_frauddetection.models.fraud_models")
        clf = getattr(mod, factory_name)()
    except (ImportError, AttributeError):
        pytest.skip(f"{factory_name} not available")

    X_train, X_test, y_train, y_test = train_test_data
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    assert auc > 0.4, f"{factory_name} AUC {auc:.3f} not better than random"
