"""Add causal, per-farm time features and future moisture targets."""

import argparse

import numpy as np
import pandas as pd

from ml_common import (
    CLEANED_DATA_PATH,
    FARM_COLUMN,
    FORECAST_DATA_PATH,
    HORIZONS,
    LAG_FEATURES,
    TARGET_COLUMN,
    TARGET_COLUMNS,
    TARGET_TIME_COLUMNS,
    TIMESTAMP_COLUMN,
    read_cleaned_dataset,
)


def build_forecasting_features(frame):
    """Create lagged features and exact 1h/3h/6h targets within each farm."""
    frame = frame.sort_values([FARM_COLUMN, TIMESTAMP_COLUMN]).copy()
    timestamps = frame[TIMESTAMP_COLUMN]
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    day_of_year = timestamps.dt.day_of_year

    frame["Current_Moisture_Pct"] = frame[TARGET_COLUMN]
    frame["Hour"] = timestamps.dt.hour
    frame["Day"] = timestamps.dt.day
    frame["Month"] = timestamps.dt.month
    frame["DayOfWeek"] = timestamps.dt.day_of_week
    frame["DayOfYear"] = day_of_year
    frame["Hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    frame["Hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    frame["DayOfYear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["DayOfYear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    grouped_moisture = frame.groupby(FARM_COLUMN, sort=False)[TARGET_COLUMN]
    for lag in (1, 4, 12, 24):
        frame[f"Moisture_Lag_{lag}"] = grouped_moisture.shift(lag)

    for window in (4, 12, 24):
        prior_readings = grouped_moisture.shift(1)
        frame[f"Moisture_Rolling_Mean_{window}"] = (
            prior_readings.groupby(frame[FARM_COLUMN], sort=False)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        frame[f"Moisture_Rolling_Std_{window}"] = (
            prior_readings.groupby(frame[FARM_COLUMN], sort=False)
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )

    for horizon, steps in HORIZONS.items():
        target_column = TARGET_COLUMNS[horizon]
        target_time_column = TARGET_TIME_COLUMNS[horizon]
        frame[target_column] = grouped_moisture.shift(-steps)
        frame[target_time_column] = frame.groupby(
            FARM_COLUMN, sort=False
        )[TIMESTAMP_COLUMN].shift(-steps)

        expected_delta = pd.Timedelta(minutes=15 * steps)
        actual_delta = frame[target_time_column] - frame[TIMESTAMP_COLUMN]
        invalid_target = actual_delta != expected_delta
        frame.loc[invalid_target, target_column] = np.nan
        frame.loc[invalid_target, target_time_column] = pd.NaT

    complete_history = frame[LAG_FEATURES].notna().all(axis=1)
    frame = frame.loc[complete_history].copy()
    return frame


def prepare_data():
    if not CLEANED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_DATA_PATH}. "
            "Run backend/clean_dataset.py first."
        )

    frame = read_cleaned_dataset()
    prepared = build_forecasting_features(frame)
    if prepared.empty:
        raise ValueError("No rows have enough per-farm history for forecasting.")

    FORECAST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(
        FORECAST_DATA_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )

    print(f"Input cleaned rows: {len(frame)}")
    print(f"Rows with at least 24 prior farm readings: {len(prepared)}")
    for horizon, target_column in TARGET_COLUMNS.items():
        valid = int(prepared[target_column].notna().sum())
        print(
            f"{horizon} forecast: {valid} valid rows; "
            f"{len(prepared) - valid} rows excluded for missing/non-15-minute target"
        )
    print("Feature families: farm ID, timestamp calendar cycles, current weather/"
          "sensor readings, per-farm past moisture lags and rolling statistics.")
    print("Targets are future moisture only; lag and rolling features use past readings.")
    print(f"Forecasting dataset saved to {FORECAST_DATA_PATH}")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        prepare_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Forecast preparation failed: {error}\n")


if __name__ == "__main__":
    main()
