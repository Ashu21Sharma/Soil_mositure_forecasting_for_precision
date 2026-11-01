"""Evaluate saved time-ordered forecasts against persistence."""

import argparse

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_common import FARM_COLUMN, FORECAST_DATA_PATH, HORIZONS, REPORTS_DIR, TIMESTAMP_COLUMN


PREDICTIONS_PATH = FORECAST_DATA_PATH.parent / "test_predictions.csv"
MODEL_ORDER = (
    "Persistence_Baseline",
    "Linear_Regression",
    "Decision_Tree",
    "Random_Forest",
)


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def evaluate_predictions():
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Test predictions not found: {PREDICTIONS_PATH}. "
            "Run backend/train_model.py first."
        )
    predictions = pd.read_csv(
        PREDICTIONS_PATH, parse_dates=[TIMESTAMP_COLUMN, "Target_Timestamp"]
    )
    required = {
        "Forecast_Horizon",
        "Model",
        TIMESTAMP_COLUMN,
        FARM_COLUMN,
        "Actual_Moisture_Pct",
        "Predicted_Moisture_Pct",
        "Persistence_Prediction_Pct",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("No test predictions are available for evaluation.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    summary_lines = [
        "Soil moisture forecasting evaluation",
        "Metrics use the chronological held-out test period only.",
        "MAE and RMSE are in volumetric-moisture percentage points; lower is better.",
        "R-squared is also shown and can be negative for weak forecasts.",
        "Persistence predicts the current measured moisture at the forecast origin.",
        "",
    ]

    for horizon in HORIZONS:
        horizon_rows = predictions.loc[
            predictions["Forecast_Horizon"] == horizon
        ].copy()
        if horizon_rows.empty:
            raise ValueError(f"No predictions found for horizon {horizon}.")
        key_columns = [TIMESTAMP_COLUMN, FARM_COLUMN]
        baseline_rows = horizon_rows.drop_duplicates(subset=key_columns)
        actual = baseline_rows["Actual_Moisture_Pct"]
        baseline = baseline_rows["Persistence_Prediction_Pct"]
        baseline_metrics = calculate_metrics(actual, baseline)
        results.append(
            {
                "Forecast_Horizon": horizon,
                "Model": "Persistence_Baseline",
                **baseline_metrics,
                "Test_Samples": len(baseline_rows),
            }
        )

        model_results = []
        for model_name, model_rows in horizon_rows.groupby("Model", sort=False):
            model_rows = model_rows.sort_values(key_columns)
            actual_rows = model_rows["Actual_Moisture_Pct"]
            predicted_rows = model_rows["Predicted_Moisture_Pct"]
            metrics = calculate_metrics(actual_rows, predicted_rows)
            result = {
                "Forecast_Horizon": horizon,
                "Model": model_name,
                **metrics,
                "Test_Samples": len(model_rows),
            }
            results.append(result)
            model_results.append(result)

        if not model_results:
            raise ValueError(f"No ML model predictions found for horizon {horizon}.")
        best_ml = min(model_results, key=lambda row: (row["MAE"], row["RMSE"]))
        best_overall = min(
            [results[-len(model_results) - 1], *model_results],
            key=lambda row: (row["MAE"], row["RMSE"]),
        )
        summary_lines.extend(
            [
                f"{horizon} forecast ({HORIZONS[horizon]} records ahead):",
                (
                    f"  Best ML model: {best_ml['Model']} "
                    f"(MAE={best_ml['MAE']:.4f}, RMSE={best_ml['RMSE']:.4f}, "
                    f"R2={best_ml['R2']:.4f})"
                ),
                (
                    f"  Persistence baseline: MAE={baseline_metrics['MAE']:.4f}, "
                    f"RMSE={baseline_metrics['RMSE']:.4f}, "
                    f"R2={baseline_metrics['R2']:.4f}"
                ),
                (
                    f"  Best overall by MAE then RMSE: {best_overall['Model']}."
                ),
                (
                    "  The selected ML model beats persistence on both MAE and "
                    "RMSE: "
                    f"{best_ml['MAE'] < baseline_metrics['MAE'] and best_ml['RMSE'] < baseline_metrics['RMSE']}."
                ),
                "",
            ]
        )

    comparison = pd.DataFrame(results)
    comparison["Model_Order"] = comparison["Model"].map(
        {name: index for index, name in enumerate(MODEL_ORDER)}
    )
    comparison = comparison.sort_values(
        ["Forecast_Horizon", "Model_Order"]
    ).drop(columns="Model_Order")
    output_csv = REPORTS_DIR / "model_comparison.csv"
    output_txt = REPORTS_DIR / "model_evaluation.txt"
    comparison.to_csv(output_csv, index=False, float_format="%.6f")
    summary_lines.append("Full comparison table:")
    summary_lines.append(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    output_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for horizon in HORIZONS:
        horizon_text = next(
            line for line in summary_lines if line.startswith(f"{horizon} forecast")
        )
        print(f"\n{horizon_text}")
        start = summary_lines.index(horizon_text)
        print("\n".join(summary_lines[start + 1 : start + 5]))
    print(f"\nResults saved to {output_csv}")
    print(f"Explanation saved to {output_txt}")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        evaluate_predictions()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Model evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
