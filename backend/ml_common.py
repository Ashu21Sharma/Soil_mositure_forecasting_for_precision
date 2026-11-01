"""Shared paths and settings for the soil-moisture ML workflow."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
HISTORICAL_FALLBACK = (
    PROJECT_ROOT / "data" / "hardware" / "historical_agricultural_dataset"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_soil_moisture.csv"
FORECAST_DATA_PATH = PROCESSED_DIR / "forecasting_features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TIMESTAMP_COLUMN = "Timestamp"
FARM_COLUMN = "Farm_ID"
TARGET_COLUMN = "Volumetric_Moisture_Pct"
HORIZONS = {
    "1h": 4,
    "3h": 12,
    "6h": 24,
}
EXPECTED_INTERVAL = pd.Timedelta(minutes=15)
TARGET_COLUMNS = {name: f"Target_Moisture_{name}" for name in HORIZONS}
TARGET_TIME_COLUMNS = {name: f"Target_Timestamp_{name}" for name in HORIZONS}

AVAILABLE_SENSOR_FEATURES = [
    "Sensor_Board_Temperature_C",
    "Raw_Capacitive_ADC",
    "Sensor_Voltage_V",
    "Atm_Pressure_inHg",
    "Atm_Temperature_C",
    "Soil_Temperature_C",
]
TIME_FEATURES = [
    "Hour",
    "Day",
    "Month",
    "DayOfWeek",
    "DayOfYear",
    "Hour_sin",
    "Hour_cos",
    "DayOfYear_sin",
    "DayOfYear_cos",
]
LAG_FEATURES = [
    "Moisture_Lag_1",
    "Moisture_Lag_4",
    "Moisture_Lag_12",
    "Moisture_Lag_24",
    "Moisture_Rolling_Mean_4",
    "Moisture_Rolling_Mean_12",
    "Moisture_Rolling_Mean_24",
    "Moisture_Rolling_Std_4",
    "Moisture_Rolling_Std_12",
    "Moisture_Rolling_Std_24",
]
FEATURE_COLUMNS = [
    FARM_COLUMN,
    "Current_Moisture_Pct",
    *TIME_FEATURES,
    *AVAILABLE_SENSOR_FEATURES,
    *LAG_FEATURES,
]


def find_source_dataset(input_path=None):
    """Prefer a historical CSV in data/raw; use the known legacy path otherwise."""
    if input_path:
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input dataset does not exist: {source}")
        return source

    if RAW_DATA_DIR.is_dir():
        for source in sorted(RAW_DATA_DIR.glob("*.csv")):
            try:
                columns = pd.read_csv(source, nrows=0).columns
            except (OSError, pd.errors.ParserError, UnicodeError):
                continue
            if TARGET_COLUMN in columns and FARM_COLUMN in columns:
                return source

    if HISTORICAL_FALLBACK.is_file():
        print(
            "NOTICE: data/raw/ has no matching historical CSV. Using the existing "
            f"historical dataset at {HISTORICAL_FALLBACK.relative_to(PROJECT_ROOT)}. "
            "The Wokwi-generated dataset is not used."
        )
        return HISTORICAL_FALLBACK

    raise FileNotFoundError(
        "No historical dataset found. Place the original CSV in data/raw/ or pass "
        "its path with --input."
    )


def read_cleaned_dataset(path=CLEANED_DATA_PATH):
    """Read the processed dataset and verify its required columns."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}. Run backend/clean_dataset.py first."
        )
    frame = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    required = {TIMESTAMP_COLUMN, FARM_COLUMN, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cleaned dataset is missing columns: {sorted(missing)}")
    frame = frame.sort_values([FARM_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    return frame


def available_feature_columns(frame):
    """Return model input columns that exist in the engineered dataset."""
    return [column for column in FEATURE_COLUMNS if column in frame.columns]
