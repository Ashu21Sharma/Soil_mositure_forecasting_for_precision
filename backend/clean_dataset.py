"""Inspect and clean the historical soil-moisture dataset without changing it."""

import argparse

import numpy as np
import pandas as pd

from ml_common import (
    CLEANED_DATA_PATH,
    FARM_COLUMN,
    REPORTS_DIR,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    find_source_dataset,
)


VALID_RANGES = {
    TARGET_COLUMN: (0, 100),
    "Raw_Capacitive_ADC": (0, 4095),
    "Sensor_Voltage_V": (0, 3.3),
    "Atm_Pressure_inHg": (20, 35),
    "Sensor_Board_Temperature_C": (-50, 80),
    "Atm_Temperature_C": (-50, 80),
    "Soil_Temperature_C": (-50, 80),
    "SM_Aggregate_Pct": (0, 100),
    "SM_Volume_Pct": (0, 100),
    "SM_Level_15cm": (0, 100),
    "SM_Volume_15cm": (0, 100),
}


def inspect_dataset(frame, report_lines):
    """Print and record the requested dataset profile."""
    details = [
        f"Dataset shape: {frame.shape}",
        f"Columns: {list(frame.columns)}",
        "Data types:",
        frame.dtypes.to_string(),
        "Missing values by column:",
        frame.isna().sum().to_string(),
        f"Exact duplicate rows: {int(frame.duplicated().sum())}",
        "Basic numeric statistics:",
        frame.describe(include="number").to_string(),
    ]
    report_lines.extend(details)
    print("\n".join(details))
    print(f"Farm identifier: {FARM_COLUMN}")
    print(f"Forecast target: {TARGET_COLUMN}")


def detect_iqr_outliers(frame):
    """Summarize IQR outliers without removing any observations."""
    summaries = []
    numeric_columns = frame.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        spread = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * spread
        upper_bound = third_quartile + 1.5 * spread
        count = int(((values < lower_bound) | (values > upper_bound)).sum())
        if count:
            summaries.append(
                f"{column}: {count} value(s) outside "
                f"[{lower_bound:.4f}, {upper_bound:.4f}]"
            )
    return summaries


def clean_dataset(source_path):
    """Clean core fields, mark impossible sensor readings missing, and save."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source_path)
    required = {TIMESTAMP_COLUMN, FARM_COLUMN, TARGET_COLUMN}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "The historical dataset must include "
            f"{sorted(missing_columns)}; found {list(frame.columns)}"
        )

    report_lines = [
        f"Source dataset: {source_path}",
        "Source is read-only; the original file is not modified.",
        "Wokwi hardware-generated readings are not used.",
        "",
    ]
    inspect_dataset(frame, report_lines)

    original_rows = len(frame)
    original_farm_ids = sorted(frame[FARM_COLUMN].dropna().astype(str).unique())
    farm_mapping = {
        farm_id: f"Farm_{index}"
        for index, farm_id in enumerate(original_farm_ids, start=1)
    }
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], errors="coerce"
    )
    for column in frame.columns:
        if column not in (TIMESTAMP_COLUMN, FARM_COLUMN):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_counts = {}
    for column, (minimum, maximum) in VALID_RANGES.items():
        if column not in frame.columns:
            continue
        invalid = frame[column].notna() & ~frame[column].between(minimum, maximum)
        invalid_counts[column] = int(invalid.sum())
        frame.loc[invalid, column] = np.nan

    missing_timestamp = int(frame[TIMESTAMP_COLUMN].isna().sum())
    missing_farm = int(frame[FARM_COLUMN].isna().sum())
    if frame[FARM_COLUMN].dtype == object:
        missing_farm += int(frame[FARM_COLUMN].astype(str).str.strip().eq("").sum())
        frame[FARM_COLUMN] = frame[FARM_COLUMN].astype("string").str.strip()
        frame.loc[frame[FARM_COLUMN].eq(""), FARM_COLUMN] = pd.NA

    invalid_target = int(frame[TARGET_COLUMN].isna().sum())
    missing_core = (
        frame[TIMESTAMP_COLUMN].isna()
        | frame[FARM_COLUMN].isna()
        | frame[TARGET_COLUMN].isna()
    )
    removed_core = int(missing_core.sum())
    frame = frame.loc[~missing_core].copy()
    frame[FARM_COLUMN] = frame[FARM_COLUMN].map(farm_mapping)
    frame = frame.sort_values([FARM_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    outliers = detect_iqr_outliers(frame)
    interval_summaries = []
    for farm_id, farm_rows in frame.groupby(FARM_COLUMN, sort=True):
        intervals = farm_rows[TIMESTAMP_COLUMN].diff().dropna()
        irregular = intervals[intervals != pd.Timedelta(minutes=15)]
        gaps = intervals[intervals > pd.Timedelta(minutes=15)]
        interval_summaries.append(
            f"{farm_id}: {len(farm_rows)} rows; "
            f"{len(irregular)} interval(s) not 15 minutes; "
            f"{len(gaps)} gap(s) longer than 15 minutes"
        )

    cleaning_details = [
        "",
        "Cleaning results:",
        f"Exact duplicate rows removed: {duplicate_count}",
        f"Rows with invalid timestamps: {missing_timestamp}",
        f"Rows with missing farm IDs: {missing_farm}",
        f"Rows with missing/invalid forecast target: {invalid_target}",
        f"Rows removed for invalid core fields: {removed_core}",
        f"Rows saved: {len(frame)} (started with {original_rows})",
        "Farm IDs normalized for model inputs:",
        *(
            [f"{old_id} -> {new_id}" for old_id, new_id in farm_mapping.items()]
            or ["No farm IDs found."]
        ),
        "Timestamp was parsed as datetime and rows sorted by Farm_ID, then Timestamp.",
        "Cleaned data types:",
        frame.dtypes.to_string(),
        "Impossible/out-of-range sensor values set to missing:",
        *(
            [
                f"{column}: {count}"
                for column, count in invalid_counts.items()
                if count
            ]
            or ["None found."]
        ),
        "Remaining missing values (numeric model features are imputed using "
        "training data only):",
        frame.isna().sum().to_string(),
        "15-minute interval check, separately per farm:",
        *(interval_summaries or ["No farms found."]),
        "IQR outliers detected, retained in the dataset:",
        *(outliers or ["None found."]),
        "Outliers are reported only; valid extreme readings are not deleted.",
    ]
    report_lines.extend(cleaning_details)

    frame.to_csv(CLEANED_DATA_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    report_path = REPORTS_DIR / "data_quality_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(cleaning_details))
    print(f"\nCleaned dataset saved to {CLEANED_DATA_PATH}")
    print(f"Quality report saved to {report_path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help=(
            "Optional source CSV path. By default uses a suitable CSV in data/raw, "
            "then the existing historical dataset path if none is present."
        ),
    )
    args = parser.parse_args()
    try:
        source = find_source_dataset(args.input)
        clean_dataset(source)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Dataset cleaning failed: {error}\n")


if __name__ == "__main__":
    main()
