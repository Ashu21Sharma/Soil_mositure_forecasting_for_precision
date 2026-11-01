"""Project data access, saved-model inference, and Wokwi collector controls."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_soil_moisture.csv"
FORECAST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecasting_features.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
HARDWARE_DATA_PATH = PROJECT_ROOT / "data" / "hardware" / "hardware_generated_dataset.csv"
COLLECTOR_PATH = PROJECT_ROOT / "backend" / "collect_hardware_data.py"
MODELS_DIR = PROJECT_ROOT / "models"

TIMESTAMP_COLUMN = "Timestamp"
FARM_COLUMN = "Farm_ID"
MOISTURE_COLUMN = "Volumetric_Moisture_Pct"
HORIZONS = ("1h", "3h", "6h")
DRY_THRESHOLD = 30
WET_THRESHOLD = 70
MODEL_FILES = {
    "Linear Regression": "linear_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
}
MODEL_NAMES = {
    "Linear_Regression": "Linear Regression",
    "Decision_Tree": "Decision Tree",
    "Random_Forest": "Random Forest",
    "Persistence_Baseline": "Naive Persistence",
}
MODEL_INPUTS = {
    "Current_Moisture_Pct": {
        "label": "Current soil moisture (%)",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "format": "%.1f",
    },
    "Soil_Temperature_C": {
        "label": "Soil temperature (°C)",
        "min": -50.0,
        "max": 80.0,
        "step": 0.1,
        "format": "%.1f",
    },
    "Atm_Temperature_C": {
        "label": "Air temperature (°C)",
        "min": -50.0,
        "max": 80.0,
        "step": 0.1,
        "format": "%.1f",
    },
    "Sensor_Board_Temperature_C": {
        "label": "Sensor board temperature (°C)",
        "min": -50.0,
        "max": 80.0,
        "step": 0.1,
        "format": "%.1f",
    },
    "Raw_Capacitive_ADC": {
        "label": "Raw capacitive ADC",
        "min": 0.0,
        "max": 4095.0,
        "step": 1.0,
        "format": "%.0f",
    },
    "Sensor_Voltage_V": {
        "label": "Sensor voltage (V)",
        "min": 0.0,
        "max": 3.3,
        "step": 0.001,
        "format": "%.3f",
    },
    "Atm_Pressure_inHg": {
        "label": "Atmospheric pressure (inHg)",
        "min": 20.0,
        "max": 35.0,
        "step": 0.01,
        "format": "%.2f",
    },
}


def load_csv(path_string, modified_time):
    """Load an uncached CSV; cache policy is provided by the Streamlit caller."""
    del modified_time
    return pd.read_csv(path_string)


def load_frame(path):
    """Load a required project CSV or raise a clear file error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required project file not found: {path}")
    return pd.read_csv(path)


def load_model_artifact(path_string, modified_time):
    """Load a saved joblib artifact; does not train or modify models."""
    del modified_time
    return joblib.load(path_string)


def latest_forecast_row(forecast_frame, farm_id):
    """Find the selected farm's latest prepared causal feature row."""
    farm_rows = forecast_frame.loc[forecast_frame[FARM_COLUMN] == farm_id].copy()
    if farm_rows.empty:
        raise ValueError(f"No prepared forecasting rows are available for {farm_id}.")
    farm_rows[TIMESTAMP_COLUMN] = pd.to_datetime(
        farm_rows[TIMESTAMP_COLUMN], errors="coerce"
    )
    farm_rows = farm_rows.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(
        TIMESTAMP_COLUMN
    )
    if farm_rows.empty:
        raise ValueError(f"No valid forecast timestamps are available for {farm_id}.")
    return farm_rows.iloc[-1]


def predict_horizons(latest_row, model_name, overrides, artifact_loader):
    """Run all forecast horizons with their saved training pipelines."""
    predictions = {}
    for horizon in HORIZONS:
        artifact_path = MODELS_DIR / (
            f"{horizon}_{MODEL_FILES[model_name]}.joblib"
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Trained model artifact not found: {artifact_path}"
            )
        artifact = artifact_loader(
            str(artifact_path), artifact_path.stat().st_mtime_ns
        )
        features = artifact.get("feature_columns")
        pipeline = artifact.get("pipeline")
        if not features or pipeline is None:
            raise ValueError(
                f"Saved artifact {artifact_path.name} lacks its model pipeline "
                "or training feature list."
            )
        missing = [feature for feature in features if feature not in latest_row.index]
        if missing:
            raise ValueError(
                f"The saved {horizon} model requires missing training features: "
                f"{missing}"
            )
        values = {feature: latest_row[feature] for feature in features}
        for feature, value in (overrides or {}).items():
            if feature in values:
                values[feature] = value
        prediction = float(
            pipeline.predict(pd.DataFrame([values], columns=features))[0]
        )
        predictions[horizon] = {
            "moisture": prediction,
            "timestamp": pd.Timestamp(latest_row[TIMESTAMP_COLUMN])
            + pd.Timedelta(hours=int(horizon[:-1])),
        }
    return predictions


def collector_log_tail(log_path, max_lines=12):
    """Return recent collector output to show connection and write errors."""
    if not log_path:
        return []
    path = Path(log_path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        return log_file.read().splitlines()[-max_lines:]


def start_collector():
    """Start the existing collector without launching or controlling Wokwi."""
    if not COLLECTOR_PATH.is_file():
        raise FileNotFoundError(f"Hardware collector not found: {COLLECTOR_PATH}")
    virtual_env_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    python_executable = (
        str(virtual_env_python) if virtual_env_python.is_file() else sys.executable
    )
    log_descriptor, log_path = tempfile.mkstemp(
        prefix="soil_wokwi_collector_", suffix=".log"
    )
    try:
        with os.fdopen(log_descriptor, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [python_executable, str(COLLECTOR_PATH)],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
    except OSError:
        Path(log_path).unlink(missing_ok=True)
        raise
    return process, log_path


def stop_collector(process, timeout=5):
    """Stop only a collector process started by this dashboard session."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def read_hardware_csv():
    """Read the real saved Wokwi file; never generate fallback readings."""
    if not HARDWARE_DATA_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(HARDWARE_DATA_PATH)
