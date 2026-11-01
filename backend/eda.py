"""Create exploratory plots for the cleaned historical soil-moisture data."""

import argparse
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from ml_common import (
    FIGURES_DIR,
    FARM_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    read_cleaned_dataset,
)


sns.set_theme(style="whitegrid")


def save_figures():
    frame = read_cleaned_dataset()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data=frame, x=TARGET_COLUMN, hue=FARM_COLUMN, bins=40, kde=True, ax=axis)
    axis.set(title="Soil Moisture Distribution", xlabel="Volumetric moisture (%)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "soil_moisture_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for farm_id, farm_rows in frame.groupby(FARM_COLUMN, sort=True):
        axis.plot(
            farm_rows[TIMESTAMP_COLUMN],
            farm_rows[TARGET_COLUMN],
            linewidth=0.8,
            label=str(farm_id),
        )
    axis.set(
        title="Soil Moisture Over Time by Farm",
        xlabel="Timestamp",
        ylabel="Volumetric moisture (%)",
    )
    axis.legend(title="Farm")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "soil_moisture_over_time.png", dpi=150)
    plt.close(figure)

    temperature_column = next(
        (
            column
            for column in (
                "Atm_Temperature_C",
                "Sensor_Board_Temperature_C",
                "Soil_Temperature_C",
            )
            if column in frame.columns
        ),
        None,
    )
    if temperature_column is None:
        raise ValueError("No temperature column is available for the EDA plot.")

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=frame,
        x=temperature_column,
        y=TARGET_COLUMN,
        hue=FARM_COLUMN,
        alpha=0.35,
        s=18,
        ax=axis,
    )
    axis.set(
        title="Temperature Versus Soil Moisture",
        xlabel=f"{temperature_column} (°C)",
        ylabel="Volumetric moisture (%)",
    )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "temperature_vs_soil_moisture.png", dpi=150)
    plt.close(figure)

    numeric = frame.select_dtypes(include="number")
    figure, axis = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        numeric.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.25,
        ax=axis,
    )
    axis.set_title("Correlation Heatmap (Numeric Columns)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(figure)

    numeric_columns = list(numeric.columns)
    column_count = 4
    row_count = math.ceil(len(numeric_columns) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16, max(4, row_count * 3)),
        squeeze=False,
    )
    for axis, column in zip(axes.flat, numeric_columns):
        sns.boxplot(y=frame[column], ax=axis, color="steelblue")
        axis.set_title(column, fontsize=9)
        axis.set_xlabel("")
    for axis in list(axes.flat)[len(numeric_columns) :]:
        axis.set_visible(False)
    figure.suptitle("Boxplots of Numeric Dataset Features", y=1.01)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "numeric_features_boxplots.png", dpi=150)
    plt.close(figure)

    files = [
        "soil_moisture_distribution.png",
        "soil_moisture_over_time.png",
        "temperature_vs_soil_moisture.png",
        "correlation_heatmap.png",
        "numeric_features_boxplots.png",
    ]
    for filename in files:
        print(f"Saved {FIGURES_DIR / filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        save_figures()
    except (OSError, ValueError) as error:
        parser.exit(1, f"EDA failed: {error}\n")


if __name__ == "__main__":
    main()
