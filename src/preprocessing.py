from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

TARGET_COLUMN = "cardio"
ID_COLUMN = "id"

RAW_FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

NUMERIC_FEATURES = [
    "age",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
]

CATEGORICAL_FEATURES = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

AGE_LIMITS_YEARS = (18, 100)

CLEANING_LIMITS = {
    "height": (120, 220),
    "weight": (35, 250),
    "ap_hi": (80, 250),
    "ap_lo": (40, 150),
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {ID_COLUMN, TARGET_COLUMN, *RAW_FEATURES}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    integer_columns = [
        ID_COLUMN,
        "age",
        "gender",
        "height",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        TARGET_COLUMN,
    ]

    for column in integer_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["weight"] = pd.to_numeric(cleaned["weight"], errors="coerce")

    cleaned = cleaned.dropna(subset=[*integer_columns, "weight"])

    cleaned[integer_columns] = cleaned[integer_columns].astype("int64")
    cleaned["weight"] = cleaned["weight"].astype("float64")

    age_years = cleaned["age"] / 365.25

    valid_mask = (
        age_years.between(*AGE_LIMITS_YEARS)
        & cleaned["height"].between(*CLEANING_LIMITS["height"])
        & cleaned["weight"].between(*CLEANING_LIMITS["weight"])
        & cleaned["ap_hi"].between(*CLEANING_LIMITS["ap_hi"])
        & cleaned["ap_lo"].between(*CLEANING_LIMITS["ap_lo"])
        & (cleaned["ap_hi"] > cleaned["ap_lo"])
        & cleaned["gender"].isin([1, 2])
        & cleaned["cholesterol"].isin([1, 2, 3])
        & cleaned["gluc"].isin([1, 2, 3])
        & cleaned["smoke"].isin([0, 1])
        & cleaned["alco"].isin([0, 1])
        & cleaned["active"].isin([0, 1])
        & cleaned[TARGET_COLUMN].isin([0, 1])
    )

    cleaned = cleaned.loc[valid_mask].reset_index(drop=True)

    return cleaned


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return x, y


def make_train_val_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_val_size = val_size / (1 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
