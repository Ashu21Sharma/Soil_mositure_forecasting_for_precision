"""Train chronological soil-moisture forecast models for 1h, 3h, and 6h."""

import argparse
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from ml_common import (
    FARM_COLUMN,
    FEATURE_COLUMNS,
    FORECAST_DATA_PATH,
    HORIZONS,
    MODELS_DIR,
    TARGET_COLUMNS,
    TARGET_TIME_COLUMNS,
    TIMESTAMP_COLUMN,
)


MODEL_CLASSES = {
    "Linear_Regression": LinearRegression,
    "Decision_Tree": DecisionTreeRegressor,
    "Random_Forest": RandomForestRegressor,
}


def load_forecasting_data():
    if not FORECAST_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Forecasting data not found: {FORECAST_DATA_PATH}. "
            "Run backend/prepare_forecasting_data.py first."
        )
    date_columns = [
        TIMESTAMP_COLUMN,
        *TARGET_TIME_COLUMNS.values(),
    ]
    frame = pd.read_csv(FORECAST_DATA_PATH, parse_dates=date_columns)
    if frame.empty:
        raise ValueError("The forecasting dataset is empty.")
    return frame


def chronological_split(frame):
    """Split on a global time cutoff and purge training labels crossing it."""
    unique_times = pd.Index(frame[TIMESTAMP_COLUMN].dropna().unique()).sort_values()
    if len(unique_times) < 5:
        raise ValueError("At least five unique timestamps are needed for a split.")
    cutoff_index = int(len(unique_times) * 0.8)
    cutoff_index = min(max(cutoff_index, 1), len(unique_times) - 1)
    cutoff = pd.Timestamp(unique_times[cutoff_index])
    return cutoff


def make_model(model_name, feature_columns):
    categorical_columns = [FARM_COLUMN]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Linear_Regression":
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "farm",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    if model_name == "Decision_Tree":
        estimator = DecisionTreeRegressor(random_state=42, min_samples_leaf=5)
    elif model_name == "Random_Forest":
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        estimator = LinearRegression()
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def train_models():
    frame = load_forecasting_data()
    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    missing_essentials = {
        FARM_COLUMN,
        "Current_Moisture_Pct",
        "Moisture_Lag_1",
    } - set(feature_columns)
    if missing_essentials:
        raise ValueError(
            f"Forecasting data lacks required input features: {sorted(missing_essentials)}"
        )

    cutoff = chronological_split(frame)
    timestamp = frame[TIMESTAMP_COLUMN]
    train_time = timestamp < cutoff
    test_time = timestamp >= cutoff
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows = []

    print(f"Chronological training cutoff: {cutoff}")
    print(f"Selected input features ({len(feature_columns)}): {feature_columns}")
    print("Preprocessing is fitted inside each model pipeline using training rows only.")

    for horizon, step_count in HORIZONS.items():
        target_column = TARGET_COLUMNS[horizon]
        target_time_column = TARGET_TIME_COLUMNS[horizon]
        if target_column not in frame or target_time_column not in frame:
            raise ValueError(f"Missing target columns for {horizon}: rerun preparation.")

        valid_target = frame[target_column].notna() & frame[target_time_column].notna()
        train_mask = (
            train_time
            & valid_target
            & (frame[target_time_column] < cutoff)
        )
        test_mask = test_time & valid_target
        train_rows = frame.loc[train_mask]
        test_rows = frame.loc[test_mask]
        if train_rows.empty or test_rows.empty:
            raise ValueError(
                f"{horizon} has an empty chronological train or test set "
                f"(train={len(train_rows)}, test={len(test_rows)})."
            )

        x_train = train_rows[feature_columns]
        y_train = train_rows[target_column].astype(float)
        x_test = test_rows[feature_columns]
        y_test = test_rows[target_column].astype(float)
        baseline = test_rows["Current_Moisture_Pct"].astype(float)

        print(
            f"\n{horizon} ({step_count} records ahead): "
            f"train={len(train_rows)}, test={len(test_rows)}"
        )

        for model_name in MODEL_CLASSES:
            pipeline = make_model(model_name, feature_columns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            artifact_path = MODELS_DIR / f"{horizon}_{model_name.lower()}.joblib"
            joblib.dump(
                {
                    "pipeline": pipeline,
                    "feature_columns": feature_columns,
                    "horizon": horizon,
                    "steps_ahead": step_count,
                    "split_cutoff": cutoff.isoformat(),
                    "training_rows": len(train_rows),
                    "test_rows": len(test_rows),
                },
                artifact_path,
            )
            for index, (_, row) in enumerate(test_rows.iterrows()):
                prediction_rows.append(
                    {
                        "Forecast_Horizon": horizon,
                        "Model": model_name,
                        TIMESTAMP_COLUMN: row[TIMESTAMP_COLUMN],
                        "Target_Timestamp": row[target_time_column],
                        FARM_COLUMN: row[FARM_COLUMN],
                        "Actual_Moisture_Pct": y_test.iloc[index],
                        "Predicted_Moisture_Pct": float(predictions[index]),
                        "Persistence_Prediction_Pct": baseline.iloc[index],
                        "Split_Cutoff": cutoff,
                    }
                )
            print(f"Trained and saved {model_name}: {artifact_path}")

    predictions_path = FORECAST_DATA_PATH.parent / "test_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(
        predictions_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )
    print(f"\nOut-of-time predictions saved to {predictions_path}")
    return predictions_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        train_models()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Model training failed: {error}\n")


if __name__ == "__main__":
    main()
