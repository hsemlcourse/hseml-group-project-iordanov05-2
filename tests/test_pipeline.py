import pandas as pd

from src.modeling import build_logistic_regression_baseline, evaluate_classifier
from src.preprocessing import (
    clean_data,
    make_train_val_test_split,
    split_features_target,
)


def _sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(12),
            "age": [18000, 19000, 20000, 21000, 22000, 23000] * 2,
            "gender": [1, 2] * 6,
            "height": [165, 180, 172, 160, 175, 168] * 2,
            "weight": [65.0, 90.0, 72.0, 58.0, 85.0, 70.0] * 2,
            "ap_hi": [120, 150, 130, 115, 160, 140] * 2,
            "ap_lo": [80, 95, 85, 75, 100, 90] * 2,
            "cholesterol": [1, 2, 1, 1, 3, 2] * 2,
            "gluc": [1, 1, 2, 1, 3, 1] * 2,
            "smoke": [0, 1, 0, 0, 1, 0] * 2,
            "alco": [0, 0, 0, 1, 0, 0] * 2,
            "active": [1, 0, 1, 1, 0, 1] * 2,
            "cardio": [0, 1, 0, 0, 1, 1] * 2,
        }
    )


def test_clean_data_removes_invalid_rows() -> None:
    raw = _sample_raw_data()
    raw.loc[0, "ap_hi"] = 30

    cleaned = clean_data(raw)

    assert len(cleaned) == len(raw) - 1
    assert cleaned["ap_hi"].min() >= 80
    assert set(cleaned["cardio"].unique()) <= {0, 1}


def test_clean_data_does_not_add_features() -> None:
    raw = _sample_raw_data()
    cleaned = clean_data(raw)

    assert cleaned.shape[1] == raw.shape[1]

    unexpected_columns = {
        "age_years",
        "bmi",
        "pulse_pressure",
        "mean_arterial_pressure",
        "bmi_category",
        "bp_category",
    }
    assert unexpected_columns.isdisjoint(cleaned.columns)


def test_split_features_target_excludes_id_and_target() -> None:
    cleaned = clean_data(_sample_raw_data())

    x, y = split_features_target(cleaned)

    assert "id" not in x.columns
    assert "cardio" not in x.columns
    assert len(x) == len(y) == len(cleaned)


def test_split_is_complete() -> None:
    cleaned = clean_data(_sample_raw_data())
    x, y = split_features_target(cleaned)

    x_train, x_val, x_test, y_train, y_val, y_test = make_train_val_test_split(
        x,
        y,
        test_size=0.25,
        val_size=0.25,
    )

    assert len(x_train) + len(x_val) + len(x_test) == len(cleaned)
    assert len(y_train) + len(y_val) + len(y_test) == len(cleaned)


def test_baseline_pipeline_fits_on_small_sample() -> None:
    cleaned = clean_data(_sample_raw_data())
    x, y = split_features_target(cleaned)

    model = build_logistic_regression_baseline()
    model.fit(x, y)

    metrics = evaluate_classifier(model, x, y)

    assert "roc_auc" in metrics
    assert 0 <= metrics["f1"] <= 1
