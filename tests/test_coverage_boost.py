"""Coverage boost tests — evaluation, logging, seed, data, and features modules.

Covers:
  - evaluation/metrics.py   (evaluate_binary, evaluate_multiclass, overfitting_check,
                              classification_report, regression_report)
  - logging_config.py       (setup_logging, get_logger, timer, track, get_progress)
  - utils/seed.py           (set_seed — including torch/tf import paths)
  - data/loaders.py         (load_raw, load_processed, save_processed, clean_data_lr,
                              load_data)
  - data/make_dataset.py    (process_data)
  - features/build_features.py (build_features, engineer_features_ensemble,
                                 create_4class_labels, add_features_lr,
                                 build_preprocessor)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def minimal_raw_df() -> pd.DataFrame:
    """Minimal DataFrame matching raw transaction schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "cc_num": np.random.randint(1_000_000, 9_999_999, n),
            "amt": np.abs(np.random.normal(50, 20, n)).clip(min=0.01),
            "merchant": np.random.choice(["merchant_A", "merchant_B"], n),
            "category": np.random.choice(
                ["grocery_pos", "shopping_net", "misc_net"], n
            ),
            "trans_date_trans_time": pd.date_range("2023-01-01", periods=n, freq="h"),
            "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "city_pop": np.random.randint(1000, 500_000, n),
        }
    )


@pytest.fixture
def pipeline_b_df() -> pd.DataFrame:
    """DataFrame with Pipeline B required columns."""
    np.random.seed(0)
    n = 100
    return pd.DataFrame(
        {
            "amt": np.abs(np.random.normal(50, 20, n)).clip(min=0.01),
            "category_shopping_net": np.random.choice([0, 1], n),
            "category_misc_net": np.random.choice([0, 1], n),
            "category_grocery_net": np.random.choice([0, 1], n),
            "customer_num_trans_1_day": np.random.randint(1, 10, n),
            "customer_num_trans_30_day": np.random.randint(5, 50, n),
            "trans_time_hrs": np.random.randint(0, 24, n),
        }
    )


@pytest.fixture
def pipeline_a_df() -> pd.DataFrame:
    """DataFrame with Pipeline A required columns."""
    np.random.seed(1)
    n = 100
    return pd.DataFrame(
        {
            "amt": np.abs(np.random.normal(50, 20, n)).clip(min=0.01),
            "avg_amt_per_customer": np.abs(np.random.normal(45, 15, n)).clip(min=0.01),
            "merchant_risk_30_day": np.random.uniform(0, 30, n),
            "merchant_risk_7_day": np.random.uniform(0, 30, n),
            "trans_time_is_night": np.random.choice([0, 1], n),
        }
    )


@pytest.fixture
def lr_feature_df() -> pd.DataFrame:
    """DataFrame for clean_data_lr tests."""
    np.random.seed(2)
    n = 50
    return pd.DataFrame(
        {
            "merchant_risk_30_day": np.random.uniform(0, 30, n).astype(str),
            "merchant_risk_7_day": np.random.uniform(0, 30, n).astype(str),
            "trans_time_is_night": np.random.choice(["0", "1"], n),
            "trans_date_is_weekend": np.random.choice(["0", "1"], n),
            "customer_num_trans_1_day": np.random.randint(1, 10, n).astype(str),
            "amt": np.abs(np.random.normal(50, 20, n)).astype(str),
            "avg_amt_per_customer": np.abs(np.random.normal(45, 15, n)).astype(str),
            "gender_M": np.random.choice(["True", "False"], n),
            "category_food_dining": np.random.choice(["True", "False"], n),
            "category_gas_transport": np.random.choice(["True", "False"], n),
            "dob": ["1990-01-01"] * n,
            "trans_date": ["2023-01-01"] * n,
            "trans_time": ["12:00:00"] * n,
        }
    )


# =============================================================================
# evaluation/metrics.py
# =============================================================================


class TestEvaluateBinary:
    """Tests for evaluate_binary()."""

    @pytest.fixture
    def binary_data(self):
        np.random.seed(0)
        n = 200
        y_true = pd.Series(np.random.choice([0, 1], n, p=[0.9, 0.1]))
        y_pred = y_true.copy()
        y_prob = y_true.astype(float)
        return y_true, y_pred, y_prob

    def test_returns_dict(self, binary_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_binary

        y_true, y_pred, y_prob = binary_data
        assert isinstance(evaluate_binary(y_true, y_pred, y_prob), dict)

    def test_has_required_keys(self, binary_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_binary

        y_true, y_pred, y_prob = binary_data
        result = evaluate_binary(y_true, y_pred, y_prob)
        for key in ["test_acc", "f1", "precision", "recall", "roc_auc", "avg_prec"]:
            assert key in result

    def test_values_in_range(self, binary_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_binary

        y_true, y_pred, y_prob = binary_data
        for v in evaluate_binary(y_true, y_pred, y_prob).values():
            assert 0.0 <= v <= 100.0

    def test_with_train_data(self, binary_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_binary

        y_true, y_pred, y_prob = binary_data
        result = evaluate_binary(
            y_true,
            y_pred,
            y_prob,
            y_train_true=y_true,
            y_train_pred=y_pred,
        )
        assert "train_acc" in result and "gap" in result

    def test_model_name_param(self, binary_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_binary

        y_true, y_pred, y_prob = binary_data
        assert isinstance(
            evaluate_binary(y_true, y_pred, y_prob, model_name="TestModel"), dict
        )


class TestEvaluateMulticlass:
    """Tests for evaluate_multiclass()."""

    @pytest.fixture
    def multiclass_data(self):
        np.random.seed(1)
        n = 200
        y_true = pd.Series(np.random.choice([0, 1, 2, 3], n))
        return y_true, y_true.copy()

    def test_returns_dict(self, multiclass_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        assert isinstance(evaluate_multiclass(*multiclass_data), dict)

    def test_has_acc_and_f1(self, multiclass_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        result = evaluate_multiclass(*multiclass_data)
        assert "test_acc" in result and "test_f1" in result

    def test_with_train_data(self, multiclass_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        y_true, y_pred = multiclass_data
        result = evaluate_multiclass(
            y_true, y_pred, y_train_true=y_true, y_train_pred=y_pred
        )
        assert "train_acc" in result and "gap" in result

    def test_perfect_predictions(self, multiclass_data):
        from mlops_frauddetection.evaluation.metrics import evaluate_multiclass

        assert evaluate_multiclass(*multiclass_data)["test_acc"] == pytest.approx(1.0)


class TestOverfittingCheck:
    """Tests for overfitting_check()."""

    def test_no_overfitting(self):
        from mlops_frauddetection.evaluation.metrics import overfitting_check

        assert overfitting_check(0.95, 0.94) == "No overfitting"

    def test_possible_overfitting(self):
        from mlops_frauddetection.evaluation.metrics import overfitting_check

        assert overfitting_check(0.99, 0.85) == "Possible overfitting"

    def test_custom_threshold(self):
        from mlops_frauddetection.evaluation.metrics import overfitting_check

        assert overfitting_check(0.95, 0.93, threshold=1.0) == "Possible overfitting"

    def test_returns_string(self):
        from mlops_frauddetection.evaluation.metrics import overfitting_check

        assert isinstance(overfitting_check(0.90, 0.90), str)


class TestClassificationReport:
    """Tests for classification_report() wrapper."""

    def test_returns_dict(self):
        from mlops_frauddetection.evaluation.metrics import classification_report

        y = np.array([0, 1, 0, 1])
        assert isinstance(classification_report(y, y), dict)

    def test_has_all_keys(self):
        from mlops_frauddetection.evaluation.metrics import classification_report

        y = np.array([0, 1, 0, 1])
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in classification_report(y, y)

    def test_perfect_score(self):
        from mlops_frauddetection.evaluation.metrics import classification_report

        y = np.array([0, 1, 0, 1])
        assert classification_report(y, y)["accuracy"] == pytest.approx(1.0)


class TestRegressionReport:
    """Tests for regression_report() wrapper."""

    def test_returns_dict(self):
        from mlops_frauddetection.evaluation.metrics import regression_report

        y = np.array([1.0, 2.0, 3.0])
        assert isinstance(regression_report(y, y), dict)

    def test_has_all_keys(self):
        from mlops_frauddetection.evaluation.metrics import regression_report

        y = np.array([1.0, 2.0, 3.0])
        for key in ["mae", "mse", "rmse", "r2"]:
            assert key in regression_report(y, y)

    def test_perfect_r2(self):
        from mlops_frauddetection.evaluation.metrics import regression_report

        y = np.array([1.0, 2.0, 3.0])
        result = regression_report(y, y)
        assert result["r2"] == pytest.approx(1.0)
        assert result["mae"] == pytest.approx(0.0)


# =============================================================================
# utils/seed.py
# =============================================================================


class TestSetSeedExtended:
    """Extended tests for set_seed() covering torch/tf import paths."""

    def test_sets_pythonhashseed(self):
        import os

        from mlops_frauddetection.utils.seed import set_seed

        set_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_numpy_deterministic(self):
        from mlops_frauddetection.utils.seed import set_seed

        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_not_installed_path(self):
        import sys

        from mlops_frauddetection.utils.seed import set_seed

        with patch.dict(sys.modules, {"torch": None}):
            set_seed(7)

    def test_tensorflow_not_installed_path(self):
        import sys

        from mlops_frauddetection.utils.seed import set_seed

        with patch.dict(sys.modules, {"tensorflow": None}):
            set_seed(7)


# =============================================================================
# logging_config.py
# =============================================================================


class TestLoggingConfig:
    """Tests for setup_logging, get_logger, timer, track, get_progress."""

    def test_get_logger_returns_logger(self):
        from mlops_frauddetection.logging_config import get_logger

        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_setup_logging_runs(self):
        from mlops_frauddetection.logging_config import setup_logging

        setup_logging("INFO")
        assert len(logging.getLogger().handlers) > 0

    def test_setup_logging_idempotent(self):
        from mlops_frauddetection.logging_config import setup_logging

        setup_logging("INFO")
        count = len(logging.getLogger().handlers)
        setup_logging("INFO")
        assert len(logging.getLogger().handlers) == count

    def test_setup_logging_debug_level(self):
        from mlops_frauddetection.logging_config import setup_logging

        setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_timer_context_manager(self):
        from mlops_frauddetection.logging_config import get_logger, timer

        with timer(get_logger("test.timer"), "test operation"):
            time.sleep(0.01)

    def test_timer_logs_completion(self, caplog):
        from mlops_frauddetection.logging_config import get_logger, timer

        logger = get_logger("test.timer.log")
        with caplog.at_level(logging.INFO, logger="test.timer.log"):
            with timer(logger, "my task"):
                pass
        assert any("my task" in r.message for r in caplog.records)

    def test_track_iterates_all_items(self):
        from mlops_frauddetection.logging_config import track

        assert list(track(list(range(5)), description="Testing")) == list(range(5))

    def test_get_progress_returns_progress(self):
        from rich.progress import Progress

        from mlops_frauddetection.logging_config import get_progress

        assert isinstance(get_progress(), Progress)


# =============================================================================
# data/loaders.py
# =============================================================================


class TestLoadRaw:
    """Tests for load_raw()."""

    def test_load_raw_success(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        csv = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv, index=False)
        monkeypatch.setattr(loaders, "RAW_DATA_DIR", tmp_path)
        df = loaders.load_raw("test.csv")
        assert isinstance(df, pd.DataFrame) and len(df) == 2

    def test_load_raw_file_not_found(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders
        from mlops_frauddetection.data.exceptions import DataLoaderError

        monkeypatch.setattr(loaders, "RAW_DATA_DIR", tmp_path)
        with pytest.raises(DataLoaderError):
            loaders.load_raw("nonexistent.csv")

    def test_load_raw_returns_dataframe(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        csv = tmp_path / "data.csv"
        pd.DataFrame({"x": range(5)}).to_csv(csv, index=False)
        monkeypatch.setattr(loaders, "RAW_DATA_DIR", tmp_path)
        assert isinstance(loaders.load_raw("data.csv"), pd.DataFrame)


class TestLoadProcessed:
    """Tests for load_processed()."""

    def test_load_processed_success(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        pd.DataFrame({"feat": [1, 2, 3]}).to_csv(tmp_path / "x_train.csv", index=False)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        assert len(loaders.load_processed("x_train.csv")) == 3

    def test_load_processed_file_not_found(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders
        from mlops_frauddetection.data.exceptions import DataLoaderError

        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        with pytest.raises(DataLoaderError):
            loaders.load_processed("missing.csv")


class TestSaveProcessed:
    """Tests for save_processed()."""

    def test_save_creates_file(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        path = loaders.save_processed(pd.DataFrame({"a": [1, 2]}), "output.csv")
        assert path.exists()

    def test_save_roundtrip(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        loaders.save_processed(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "test.csv")
        assert list(pd.read_csv(tmp_path / "test.csv").columns) == ["a", "b"]

    def test_save_returns_path(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        assert isinstance(
            loaders.save_processed(pd.DataFrame({"x": [1]}), "out.csv"), Path
        )


class TestCleanDataLr:
    """Tests for clean_data_lr()."""

    def test_returns_tuple_of_dataframes(self, lr_feature_df):
        from mlops_frauddetection.data.loaders import clean_data_lr

        train, test = clean_data_lr(lr_feature_df, lr_feature_df.copy())
        assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)

    def test_drops_date_columns(self, lr_feature_df):
        from mlops_frauddetection.data.loaders import clean_data_lr

        train, _ = clean_data_lr(lr_feature_df, lr_feature_df.copy())
        for col in ["dob", "trans_date", "trans_time"]:
            assert col not in train.columns

    def test_bool_columns_mapped_to_int(self, lr_feature_df):
        from mlops_frauddetection.data.loaders import clean_data_lr

        train, _ = clean_data_lr(lr_feature_df, lr_feature_df.copy())
        if "gender_M" in train.columns:
            assert train["gender_M"].dtype in [int, np.int64, np.int32]

    def test_no_nans_in_numeric_cols(self, lr_feature_df):
        from mlops_frauddetection.data.loaders import clean_data_lr

        train, _ = clean_data_lr(lr_feature_df, lr_feature_df.copy())
        for col in ["merchant_risk_30_day", "merchant_risk_7_day"]:
            if col in train.columns:
                assert train[col].isna().sum() == 0

    def test_does_not_modify_original(self, lr_feature_df):
        from mlops_frauddetection.data.loaders import clean_data_lr

        original_cols = list(lr_feature_df.columns)
        clean_data_lr(lr_feature_df, lr_feature_df.copy())
        assert list(lr_feature_df.columns) == original_cols


class TestLoadData:
    """Tests for load_data()."""

    def test_load_data_returns_four_items(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        for name in ["x_train.csv", "x_test.csv"]:
            pd.DataFrame({"feat": [1, 2]}).to_csv(tmp_path / name, index=False)
        for name in ["y_train.csv", "y_test.csv"]:
            pd.DataFrame({"is_fraud": [0, 1]}).to_csv(tmp_path / name, index=False)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        assert len(loaders.load_data(tmp_path)) == 4

    def test_load_data_shapes_match(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders

        for name in ["x_train.csv", "x_test.csv"]:
            pd.DataFrame({"a": range(10)}).to_csv(tmp_path / name, index=False)
        for name in ["y_train.csv", "y_test.csv"]:
            pd.DataFrame({"is_fraud": range(10)}).to_csv(tmp_path / name, index=False)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        X_train, y_train, X_test, y_test = loaders.load_data(tmp_path)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


# =============================================================================
# data/make_dataset.py
# =============================================================================


class TestProcessData:
    """Tests for process_data()."""

    def test_process_data_creates_output_files(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.loaders as loaders
        import mlops_frauddetection.data.make_dataset as md

        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "cc_num": np.random.randint(1_000_000, 9_999_999, n),
                "amt": np.abs(np.random.normal(50, 20, n)),
                "merchant": np.random.choice(["m_A", "m_B"], n),
                "category": np.random.choice(["grocery_pos", "shopping_net"], n),
                "trans_date_trans_time": pd.date_range(
                    "2023-01-01", periods=n, freq="h"
                ),
                "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
                "city_pop": np.random.randint(1000, 500_000, n),
            }
        )
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        df.to_csv(raw_dir / "data_100k.csv", index=False)

        monkeypatch.setattr(md, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(loaders, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", out_dir)
        monkeypatch.setattr(md, "PROCESSED_DATA_DIR", out_dir)

        md.process_data(raw_dir, "data_100k.csv", out_dir)

        for fname in ["x_train.csv", "x_test.csv", "y_train.csv", "y_test.csv"]:
            assert (out_dir / fname).exists(), f"{fname} not created"

    def test_process_data_raises_on_missing_file(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.make_dataset as md
        from mlops_frauddetection.data.exceptions import RawDataNotFoundError

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(md, "RAW_DATA_DIR", raw_dir)
        with pytest.raises(RawDataNotFoundError):
            md.process_data(raw_dir, "nonexistent.csv", tmp_path / "out")

    def test_process_data_raises_on_none_input_file(self, tmp_path, monkeypatch):
        import mlops_frauddetection.data.make_dataset as md
        from mlops_frauddetection.data.exceptions import RawDataNotFoundError

        monkeypatch.setattr(md, "RAW_DATA_DIR", tmp_path)
        with pytest.raises(RawDataNotFoundError):
            md.process_data(tmp_path, None, tmp_path / "out")


# =============================================================================
# features/build_features.py
# =============================================================================


class TestBuildFeatures:
    """Tests for build_features()."""

    def test_returns_dataframe(self, minimal_raw_df):
        from mlops_frauddetection.features.build_features import build_features

        assert isinstance(build_features(minimal_raw_df), pd.DataFrame)

    def test_does_not_modify_original(self, minimal_raw_df):
        from mlops_frauddetection.features.build_features import build_features

        original_cols = set(minimal_raw_df.columns)
        build_features(minimal_raw_df)
        assert set(minimal_raw_df.columns) == original_cols

    def test_adds_customer_txn_count(self, minimal_raw_df):
        from mlops_frauddetection.features.build_features import build_features

        assert "customer_txn_count" in build_features(minimal_raw_df).columns

    def test_adds_avg_amt_per_customer(self, minimal_raw_df):
        from mlops_frauddetection.features.build_features import build_features

        assert "avg_amt_per_customer" in build_features(minimal_raw_df).columns

    def test_adds_merchant_txn_count(self, minimal_raw_df):
        from mlops_frauddetection.features.build_features import build_features

        assert "merchant_txn_count" in build_features(minimal_raw_df).columns

    def test_no_cc_num_skips_customer_features(self):
        from mlops_frauddetection.features.build_features import build_features

        df = pd.DataFrame({"amt": [1.0, 2.0], "merchant": ["m_A", "m_B"]})
        assert "customer_txn_count" not in build_features(df).columns

    def test_no_merchant_skips_merchant_features(self):
        from mlops_frauddetection.features.build_features import build_features

        df = pd.DataFrame({"cc_num": [1, 2], "amt": [1.0, 2.0]})
        assert "merchant_txn_count" not in build_features(df).columns


class TestEngineerFeaturesEnsemble:
    """Tests for engineer_features_ensemble()."""

    def test_returns_dataframe(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        assert isinstance(engineer_features_ensemble(pipeline_b_df), pd.DataFrame)

    def test_adds_log_amt(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        assert "log_amt" in engineer_features_ensemble(pipeline_b_df).columns

    def test_adds_category_is_online(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        assert "category_is_online" in engineer_features_ensemble(pipeline_b_df).columns

    def test_adds_velocity_ratio(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        assert "velocity_ratio" in engineer_features_ensemble(pipeline_b_df).columns

    def test_adds_cyclic_hour_features(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        result = engineer_features_ensemble(pipeline_b_df)
        assert "trans_hour_sin" in result.columns and "trans_hour_cos" in result.columns

    def test_log_amt_non_negative(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        assert (engineer_features_ensemble(pipeline_b_df)["log_amt"] >= 0).all()

    def test_does_not_modify_original(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            engineer_features_ensemble,
        )

        original_cols = set(pipeline_b_df.columns)
        engineer_features_ensemble(pipeline_b_df)
        assert set(pipeline_b_df.columns) == original_cols


class TestCreate4ClassLabels:
    """Tests for create_4class_labels()."""

    def test_returns_series(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import create_4class_labels

        y = pd.Series(np.random.choice([0, 1], len(pipeline_a_df), p=[0.95, 0.05]))
        assert isinstance(create_4class_labels(pipeline_a_df, y), pd.Series)

    def test_labels_are_0_to_3(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import create_4class_labels

        y = pd.Series(np.random.choice([0, 1], len(pipeline_a_df), p=[0.95, 0.05]))
        assert set(create_4class_labels(pipeline_a_df, y).unique()).issubset(
            {0, 1, 2, 3}
        )

    def test_length_matches_input(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import create_4class_labels

        y = pd.Series(np.zeros(len(pipeline_a_df), dtype=int))
        assert len(create_4class_labels(pipeline_a_df, y)) == len(pipeline_a_df)

    def test_all_legit_gives_only_0_or_1(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import create_4class_labels

        y = pd.Series(np.zeros(len(pipeline_a_df), dtype=int))
        assert set(create_4class_labels(pipeline_a_df, y).unique()).issubset({0, 1})

    def test_all_fraud_gives_only_2_or_3(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import create_4class_labels

        y = pd.Series(np.ones(len(pipeline_a_df), dtype=int))
        assert set(create_4class_labels(pipeline_a_df, y).unique()).issubset({2, 3})


class TestAddFeaturesLr:
    """Tests for add_features_lr()."""

    def test_returns_dataframe(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import add_features_lr

        assert isinstance(add_features_lr(pipeline_a_df), pd.DataFrame)

    def test_adds_amt_ratio(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import add_features_lr

        assert "amt_ratio" in add_features_lr(pipeline_a_df).columns

    def test_adds_combined_risk(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import add_features_lr

        assert "combined_risk" in add_features_lr(pipeline_a_df).columns

    def test_adds_five_features(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import add_features_lr

        new_cols = set(add_features_lr(pipeline_a_df).columns) - set(
            pipeline_a_df.columns
        )
        assert len(new_cols) == 5

    def test_does_not_modify_original(self, pipeline_a_df):
        from mlops_frauddetection.features.build_features import add_features_lr

        original_cols = set(pipeline_a_df.columns)
        add_features_lr(pipeline_a_df)
        assert set(pipeline_a_df.columns) == original_cols


class TestBuildPreprocessor:
    """Tests for build_preprocessor()."""

    def test_returns_three_items(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            build_preprocessor,
            engineer_features_ensemble,
        )

        assert len(build_preprocessor(engineer_features_ensemble(pipeline_b_df))) == 3

    def test_preprocessor_is_column_transformer(self, pipeline_b_df):
        from sklearn.compose import ColumnTransformer

        from mlops_frauddetection.features.build_features import (
            build_preprocessor,
            engineer_features_ensemble,
        )

        preprocessor, _, _ = build_preprocessor(
            engineer_features_ensemble(pipeline_b_df)
        )
        assert isinstance(preprocessor, ColumnTransformer)

    def test_numeric_features_is_list(self, pipeline_b_df):
        from mlops_frauddetection.features.build_features import (
            build_preprocessor,
            engineer_features_ensemble,
        )

        _, numeric_features, _ = build_preprocessor(
            engineer_features_ensemble(pipeline_b_df)
        )
        assert isinstance(numeric_features, list)
