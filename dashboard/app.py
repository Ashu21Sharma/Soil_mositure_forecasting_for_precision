"""Six-page precision-agriculture forecasting dashboard."""

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from .components import (
        COLORS,
        LOGO_PATH,
        apply_dashboard_style,
        render_brand_sidebar,
        render_page_banner,
        sensor_gauge,
        style_plot,
        threshold_gauge,
    )
    from .utils import (
        CLEANED_DATA_PATH,
        COMPARISON_PATH,
        FARM_COLUMN,
        HARDWARE_DATA_PATH,
        HORIZONS,
        MODEL_FILES,
        MODEL_INPUTS,
        MODEL_NAMES,
        MOISTURE_COLUMN,
        MODELS_DIR,
        PREDICTIONS_PATH,
        TIMESTAMP_COLUMN,
        WET_THRESHOLD,
        DRY_THRESHOLD,
        FORECAST_DATA_PATH,
        collector_log_tail,
        latest_forecast_row,
        load_csv as read_csv,
        load_model_artifact,
        predict_horizons,
        start_collector,
        stop_collector,
    )
except ImportError:  # pragma: no cover - supports direct Streamlit script execution.
    from components import (
        COLORS,
        LOGO_PATH,
        apply_dashboard_style,
        render_brand_sidebar,
        render_page_banner,
        sensor_gauge,
        style_plot,
        threshold_gauge,
    )
    from utils import (
        CLEANED_DATA_PATH,
        COMPARISON_PATH,
        FARM_COLUMN,
        HARDWARE_DATA_PATH,
        HORIZONS,
        MODEL_FILES,
        MODEL_INPUTS,
        MODEL_NAMES,
        MOISTURE_COLUMN,
        MODELS_DIR,
        PREDICTIONS_PATH,
        TIMESTAMP_COLUMN,
        WET_THRESHOLD,
        DRY_THRESHOLD,
        FORECAST_DATA_PATH,
        collector_log_tail,
        latest_forecast_row,
        load_csv as read_csv,
        load_model_artifact,
        predict_horizons,
        start_collector,
        stop_collector,
    )


PAGE_OPTIONS = (
    "Dashboard Overview",
    "Soil Moisture Prediction",
    "Data Visualization",
    "Model Performance",
    "Irrigation Recommendation",
    "Wokwi Demo Data",
)
try:
    IS_CLOUD_DEPLOYMENT = (
        st.secrets.get("SOIL_MOISTURE_CLOUD_DEPLOYMENT", False) is True
    )
except StreamlitSecretNotFoundError:
    IS_CLOUD_DEPLOYMENT = False
ENVIRONMENTAL_FEATURES = {
    "Soil_Temperature_C": "Soil temperature (°C)",
    "Atm_Temperature_C": "Air temperature (°C)",
    "Sensor_Board_Temperature_C": "Sensor board temperature (°C)",
    "Atm_Pressure_inHg": "Atmospheric pressure (inHg)",
    "Sensor_Voltage_V": "Sensor voltage (V)",
}
SENSOR_MEANINGS = {
    "Volumetric_Moisture_Pct": (
        "Forecast target: volumetric soil-water content, expressed as a percentage."
    ),
    "Soil_Temperature_C": "Temperature measured in the soil/root zone (°C).",
    "Atm_Temperature_C": "Air temperature recorded in the historical dataset (°C).",
    "Sensor_Board_Temperature_C": "Temperature at the sensor board (°C).",
    "Raw_Capacitive_ADC": "Raw capacitive probe count; dielectric moisture proxy.",
    "Sensor_Voltage_V": "Sensor voltage measurement (V).",
    "Atm_Pressure_inHg": "Atmospheric pressure measurement (inHg).",
    "SM_Level_15cm": "Soil-moisture level recorded at 15 cm.",
    "SM_Volume_15cm": "Soil-moisture volume recorded at 15 cm.",
    "SM_Aggregate_Pct": "Aggregated soil-moisture percentage.",
    "SM_Volume_Pct": "Soil volumetric percentage from the source dataset.",
}

st.set_page_config(
    page_title="Soil Moisture Forecasting | Precision Agriculture",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_dashboard_style()


@st.cache_data(ttl=10)
def load_project_csv(path_string, modified_time):
    """Cache CSV reads briefly; the caller supplies the file modification time."""
    return read_csv(path_string, modified_time)


@st.cache_resource
def load_saved_model(path_string, modified_time):
    """Cache trained artifacts without fitting or modifying the models."""
    return load_model_artifact(path_string, modified_time)


def load_required_frame(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required project file not found: {path}")
    return load_project_csv(str(path), path.stat().st_mtime_ns)


def prepare_historical(frame):
    """Return a cleaned dashboard copy, never changing the source CSV."""
    result = frame.copy()
    result[FARM_COLUMN] = result[FARM_COLUMN].astype("string").str.strip()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN], errors="coerce"
    )
    numeric = result.select_dtypes(include="number").columns
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    required = [TIMESTAMP_COLUMN, FARM_COLUMN, MOISTURE_COLUMN]
    return result.dropna(subset=required).sort_values(
        [FARM_COLUMN, TIMESTAMP_COLUMN]
    )


def add_brand_header(title, subtitle):
    render_page_banner(title, subtitle)


def condition_for(moisture, low_threshold, upper_threshold):
    if moisture < low_threshold:
        return "Dry"
    if moisture > upper_threshold:
        return "Wet"
    return "Normal"


def show_overview(historical, selected_farm):
    add_brand_header(
        "Soil Moisture Forecasting for Precision Agriculture",
        "Machine Learning-Based Predictive Irrigation for Precision Agriculture.",
    )
    st.markdown(
        '<div class="overview-section-title">Field Telemetry &amp; Project Overview</div>',
        unsafe_allow_html=True,
    )
    avg_soil_temperature = (
        historical["Soil_Temperature_C"].mean()
        if "Soil_Temperature_C" in historical
        else float("nan")
    )
    metrics = st.columns(4)
    metrics[0].metric("Total Observations", f"{len(historical):,}")
    metrics[1].metric(
        "Monitored Farm Plots", f"{historical[FARM_COLUMN].nunique():,}"
    )
    metrics[2].metric(
        "Mean Soil Moisture", f"{historical[MOISTURE_COLUMN].mean():.2f}%"
    )
    metrics[3].metric(
        "Mean Soil Temperature",
        f"{avg_soil_temperature:.2f} °C"
        if pd.notna(avg_soil_temperature)
        else "Not available",
    )

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown(
            '<div class="overview-section-title">Precision Agriculture Objective</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="overview-copy">Soil-moisture forecasts help growers '
            "anticipate changing field conditions and plan irrigation rather than "
            "relying only on a current reading. This project uses farm-aware "
            "historical sensor records to predict future moisture and compare "
            "machine-learning approaches. Forecasts are decision support and "
            "should be interpreted with local agronomic knowledge.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Project objectives**\n"
            "- Predict soil moisture 1, 3, and 6 hours ahead.\n"
            "- Compare saved ML models with a naive persistence baseline.\n"
            "- Support irrigation planning using the selected forecast.\n"
            "- Keep simulator readings separate from historical farm data."
        )
        st.markdown("**Monitored sensor attributes**")
        attributes = [
            {"Feature": column, "Physical meaning": meaning}
            for column, meaning in SENSOR_MEANINGS.items()
            if column in historical.columns
        ]
        if attributes:
            st.dataframe(
                pd.DataFrame(attributes),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No sensor attribute descriptions match the cleaned CSV columns.")

    with right:
        st.markdown(
            '<div class="overview-section-title">Dataset Telemetry Window</div>',
            unsafe_allow_html=True,
        )
        start_time = historical[TIMESTAMP_COLUMN].min()
        end_time = historical[TIMESTAMP_COLUMN].max()
        intervals = (
            historical.groupby(FARM_COLUMN)[TIMESTAMP_COLUMN].diff().dropna()
        )
        median_interval = intervals.median() if not intervals.empty else pd.NaT
        if pd.notna(median_interval):
            minutes = int(median_interval.total_seconds() // 60)
            temporal_resolution = (
                f"{minutes} minutes" if minutes < 60 else f"{minutes // 60} hour(s)"
            )
        else:
            temporal_resolution = "Not available"
        st.markdown(
            "<div class='overview-telemetry'>"
            f"<strong>Start:</strong> {start_time:%Y-%m-%d %H:%M}<br>"
            f"<strong>End:</strong> {end_time:%Y-%m-%d %H:%M}<br>"
            f"<strong>Temporal resolution:</strong> {temporal_resolution}<br>"
            f"<strong>Observations:</strong> {len(historical):,}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Sample dataset records · {selected_farm}**")
        farm_sample = historical.loc[historical[FARM_COLUMN] == selected_farm]
        sample_columns = [
            column
            for column in (
                TIMESTAMP_COLUMN,
                FARM_COLUMN,
                MOISTURE_COLUMN,
                "Soil_Temperature_C",
                "Atm_Temperature_C",
            )
            if column in historical.columns
        ]
        st.dataframe(
            farm_sample[sample_columns].head(8),
            width="stretch",
            hide_index=True,
            column_config={
                TIMESTAMP_COLUMN: st.column_config.DatetimeColumn(
                    format="YYYY-MM-DD HH:mm"
                ),
                MOISTURE_COLUMN: st.column_config.NumberColumn(format="%.2f"),
                "Soil_Temperature_C": st.column_config.NumberColumn(format="%.2f"),
                "Atm_Temperature_C": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    st.subheader("Soil Moisture Distribution")
    histogram = px.histogram(
        historical,
        x=MOISTURE_COLUMN,
        color=FARM_COLUMN,
        nbins=35,
        barmode="overlay",
        opacity=0.78,
        color_discrete_sequence=["#237a57", "#86b98e"],
        labels={MOISTURE_COLUMN: "Volumetric moisture (%)"},
    )
    st.plotly_chart(
        style_plot(histogram, "Historical soil-moisture distribution"),
        width="stretch",
        key="overview_distribution",
    )


def _feature_input_value(row, name):
    if name not in row.index:
        raise ValueError(f"Latest prepared record is missing model input {name}.")
    value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
    if pd.isna(value):
        raise ValueError(
            f"Latest prepared record has no valid value for model input {name}."
        )
    return float(value)


def _prediction_form(latest_row, model_name):
    artifacts = {}
    required_inputs = set()
    for horizon in HORIZONS:
        path = MODELS_DIR / f"{horizon}_{MODEL_FILES[model_name]}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Trained model artifact not found: {path}")
        artifact = load_saved_model(str(path), path.stat().st_mtime_ns)
        features = artifact.get("feature_columns")
        if not features or artifact.get("pipeline") is None:
            raise ValueError(
                f"Model artifact {path.name} has no pipeline or feature list."
            )
        missing = [feature for feature in features if feature not in latest_row.index]
        if missing:
            raise ValueError(
                f"Prepared data is missing {horizon} model features: {missing}"
            )
        artifacts[horizon] = artifact
        required_inputs.update(features)

    editable = [name for name in MODEL_INPUTS if name in required_inputs]
    values = {}
    with st.form("soil_moisture_prediction_form"):
        st.markdown("**Current observed sensor inputs**")
        columns = st.columns(2)
        for index, name in enumerate(editable):
            spec = MODEL_INPUTS[name]
            default = _feature_input_value(latest_row, name)
            default = min(max(default, spec["min"]), spec["max"])
            values[name] = columns[index % 2].number_input(
                spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                value=default,
                step=spec["step"],
                format=spec["format"],
                key=f"predict_input_{name}",
            )
        st.caption(
            "Farm ID, calendar values, moisture lags and rolling statistics use "
            "the selected farm's actual latest prepared row. Only the displayed "
            "current sensor fields are user-editable."
        )
        submitted = st.form_submit_button("Predict Moisture", type="primary")
    return artifacts, values, submitted


def show_prediction(farm_history, forecast_frame, selected_farm, model_name, history_days):
    add_brand_header(
        "Soil Moisture Forecasting",
        "Predict future soil moisture using trained machine-learning models.",
    )
    st.markdown(
        '<p class="section-note">Predictions use saved model pipelines and the '
        "same preprocessing fitted during training; the dashboard never retrains.</p>",
        unsafe_allow_html=True,
    )
    try:
        latest_row = latest_forecast_row(forecast_frame, selected_farm)
        artifacts, user_values, submitted = _prediction_form(
            latest_row, model_name
        )
    except (OSError, KeyError, ValueError, TypeError) as error:
        st.error(f"Cannot prepare prediction inputs: {error}")
        return

    context = (selected_farm, model_name)
    if submitted:
        try:
            predictions = {}
            for horizon, artifact in artifacts.items():
                features = artifact["feature_columns"]
                input_values = {
                    feature: latest_row[feature] for feature in features
                }
                for feature, value in user_values.items():
                    if feature in input_values:
                        input_values[feature] = value
                input_frame = pd.DataFrame([input_values], columns=features)
                prediction = float(
                    artifact["pipeline"].predict(input_frame)[0]
                )
                predictions[horizon] = {
                    "moisture": prediction,
                    "timestamp": pd.Timestamp(latest_row[TIMESTAMP_COLUMN])
                    + pd.Timedelta(hours=int(horizon[:-1])),
                }
            st.session_state["prediction_result"] = {
                "context": context,
                "inputs": user_values.copy(),
                "predictions": predictions,
                "timestamp": pd.Timestamp(latest_row[TIMESTAMP_COLUMN]),
            }
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Prediction failed: {error}")
            return

    result = st.session_state.get("prediction_result")
    if not result or result["context"] != context:
        st.info("Set the input values above and select **Predict Moisture**.")
        return

    predictions = result["predictions"]
    inputs = result["inputs"]
    current_moisture = inputs.get(
        "Current_Moisture_Pct",
        _feature_input_value(latest_row, "Current_Moisture_Pct"),
    )
    st.subheader("Forecast results")
    cards = st.columns(3)
    for card, horizon in zip(cards, HORIZONS):
        predicted = predictions[horizon]["moisture"]
        card.metric(
            f"{horizon} forecast",
            f"{predicted:.2f}%",
            delta=f"{predicted - current_moisture:+.2f} percentage points",
        )
        card.caption(
            f"{predictions[horizon]['timestamp']:%Y-%m-%d %H:%M} · "
            f"{condition_for(predicted, st.session_state['dry_threshold'], st.session_state['upper_threshold'])}"
        )

    current_time = pd.Timestamp(latest_row[TIMESTAMP_COLUMN])
    history = farm_history.copy()
    history = history.loc[
        history[TIMESTAMP_COLUMN]
        >= current_time - pd.Timedelta(days=history_days)
    ].sort_values(TIMESTAMP_COLUMN)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history[TIMESTAMP_COLUMN],
            y=history[MOISTURE_COLUMN],
            name="Historical moisture",
            mode="lines",
            line={"color": COLORS["green"], "width": 2.4},
        )
    )
    forecast_times = [current_time] + [
        predictions[horizon]["timestamp"] for horizon in HORIZONS
    ]
    forecast_values = [current_moisture] + [
        predictions[horizon]["moisture"] for horizon in HORIZONS
    ]
    figure.add_trace(
        go.Scatter(
            x=forecast_times,
            y=forecast_values,
            name="Current and predicted moisture",
            mode="lines+markers",
            line={"color": COLORS["orange"], "width": 2.5, "dash": "dash"},
            marker={"size": 9},
        )
    )
    figure.add_hline(
        y=st.session_state["dry_threshold"],
        line_dash="dot",
        line_color=COLORS["red"],
        annotation_text="Dry threshold",
    )
    figure.add_hline(
        y=st.session_state["upper_threshold"],
        line_dash="dot",
        line_color=COLORS["gold"],
        annotation_text="Upper threshold",
    )
    figure.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Volumetric soil moisture (%)",
    )
    st.plotly_chart(
        style_plot(figure, f"Historical and predicted moisture · {selected_farm}", 430),
        width="stretch",
        key="prediction_forecast_chart",
    )

    input_rows = [
        {"Input feature": label, "User value": value}
        for name, value in inputs.items()
        for label in [MODEL_INPUTS[name]["label"]]
    ]
    input_rows.extend(
        [
            {
                "Input feature": f"{horizon} forecast",
                "User value": predictions[horizon]["moisture"],
            }
            for horizon in HORIZONS
        ]
    )
    st.subheader("Input vs. predicted values")
    st.dataframe(
        pd.DataFrame(input_rows),
        width="stretch",
        hide_index=True,
        column_config={"User value": st.column_config.NumberColumn(format="%.3f")},
    )


def _filter_date_range(frame, widget_key):
    minimum = frame[TIMESTAMP_COLUMN].min().date()
    maximum = frame[TIMESTAMP_COLUMN].max().date()
    selected_range = st.date_input(
        "Date range",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
        key=widget_key,
    )
    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        return frame.iloc[0:0]
    start, end = selected_range
    return frame.loc[
        frame[TIMESTAMP_COLUMN].dt.date.between(start, end)
    ]


def show_visualization(historical):
    add_brand_header(
        "Data Visualization & Exploratory Analysis",
        "Explore historical soil moisture and environmental sensor relationships.",
    )
    farms = sorted(historical[FARM_COLUMN].dropna().astype(str).unique())
    filters = st.columns(2)
    selected_farm = filters[0].selectbox(
        "Farm filter", ["All farms", *farms], key="visualization_farm"
    )
    env_columns = [
        column
        for column in ENVIRONMENTAL_FEATURES
        if column in historical.columns
    ]
    if not env_columns:
        st.error("No supported environmental variables exist in the cleaned dataset.")
        return
    environmental_feature = filters[1].selectbox(
        "Environmental variable",
        env_columns,
        format_func=lambda column: ENVIRONMENTAL_FEATURES[column],
        key="visualization_environment",
    )
    st.caption(
        "Use the date range below to filter all charts. Historical and Wokwi "
        "datasets are not combined."
    )
    base = historical.copy()
    base = base.dropna(subset=[TIMESTAMP_COLUMN, MOISTURE_COLUMN])
    date_range = _filter_date_range(base, "visualization_date_range")
    if date_range.empty:
        st.warning("No historical observations fall inside the selected date range.")
        return

    plot_data = date_range
    if selected_farm != "All farms":
        plot_data = plot_data.loc[plot_data[FARM_COLUMN] == selected_farm]
    if plot_data.empty:
        st.info("No data is available for that farm and date range.")
        return

    left, right = st.columns(2)
    with left:
        distribution = px.histogram(
            plot_data,
            x=MOISTURE_COLUMN,
            color=FARM_COLUMN if selected_farm == "All farms" else None,
            nbins=35,
            color_discrete_sequence=["#237a57", "#86b98e"],
            labels={MOISTURE_COLUMN: "Volumetric moisture (%)"},
        )
        st.plotly_chart(
            style_plot(distribution, "Soil moisture distribution"),
            width="stretch",
            key="eda_moisture_distribution",
        )
    with right:
        trend = px.line(
            plot_data,
            x=TIMESTAMP_COLUMN,
            y=MOISTURE_COLUMN,
            color=FARM_COLUMN if selected_farm == "All farms" else None,
            color_discrete_sequence=["#237a57", "#86b98e"],
            labels={
                TIMESTAMP_COLUMN: "Timestamp",
                MOISTURE_COLUMN: "Volumetric moisture (%)",
            },
        )
        st.plotly_chart(
            style_plot(trend, "Soil moisture trend over time"),
            width="stretch",
            key="eda_moisture_trend",
        )

    if environmental_feature in plot_data.columns:
        scatter = px.scatter(
            plot_data,
            x=environmental_feature,
            y=MOISTURE_COLUMN,
            color=FARM_COLUMN if selected_farm == "All farms" else None,
            color_discrete_sequence=["#237a57", "#86b98e"],
            opacity=0.65,
            labels={
                environmental_feature: ENVIRONMENTAL_FEATURES[environmental_feature],
                MOISTURE_COLUMN: "Volumetric moisture (%)",
            },
        )
        st.plotly_chart(
            style_plot(
                scatter,
                f"{ENVIRONMENTAL_FEATURES[environmental_feature]} vs soil moisture",
            ),
            width="stretch",
            key="eda_temperature_scatter",
        )

    st.subheader("Time series of environmental variables")
    env_series = plot_data.dropna(subset=[environmental_feature])
    environment_figure = px.line(
        env_series,
        x=TIMESTAMP_COLUMN,
        y=environmental_feature,
        color=FARM_COLUMN if selected_farm == "All farms" else None,
        color_discrete_sequence=["#db8b32", "#83a5bf"],
        labels={
            TIMESTAMP_COLUMN: "Timestamp",
            environmental_feature: ENVIRONMENTAL_FEATURES[environmental_feature],
        },
    )
    st.plotly_chart(
        style_plot(
            environment_figure,
            ENVIRONMENTAL_FEATURES[environmental_feature],
        ),
        width="stretch",
        key="eda_environmental_series",
    )

    if historical[FARM_COLUMN].nunique() > 1:
        st.subheader("Farm-wise comparison")
        comparison_chart = px.line(
            date_range,
            x=TIMESTAMP_COLUMN,
            y=MOISTURE_COLUMN,
            color=FARM_COLUMN,
            color_discrete_sequence=["#237a57", "#86b98e"],
            labels={
                TIMESTAMP_COLUMN: "Timestamp",
                MOISTURE_COLUMN: "Volumetric moisture (%)",
            },
        )
        st.plotly_chart(
            style_plot(comparison_chart, "Soil moisture by farm"),
            width="stretch",
            key="eda_farm_comparison",
        )


def _load_evaluation_data():
    comparison = load_required_frame(COMPARISON_PATH)
    predictions = load_required_frame(PREDICTIONS_PATH)
    return comparison, predictions


def show_model_performance():
    add_brand_header(
        "Model Evaluation & Benchmarking",
        "Compare saved model results on the chronological held-out test period.",
    )
    try:
        comparison, predictions = _load_evaluation_data()
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load saved evaluation results: {error}")
        return
    required = {"Forecast_Horizon", "Model", "MAE", "RMSE", "R2", "Test_Samples"}
    if required - set(comparison.columns):
        st.error(
            "Model comparison CSV is missing columns: "
            f"{sorted(required - set(comparison.columns))}"
        )
        return
    horizon = st.selectbox("Forecast horizon", HORIZONS, key="performance_horizon")
    results = comparison.loc[comparison["Forecast_Horizon"] == horizon].copy()
    for column in ("MAE", "RMSE", "R2", "Test_Samples"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results = results.dropna(subset=["MAE", "RMSE", "R2"])
    results["Model name"] = results["Model"].map(MODEL_NAMES).fillna(
        results["Model"]
    )
    if results.empty:
        st.warning(f"No evaluation metrics were recorded for the {horizon} horizon.")
        return

    ml_results = results.loc[results["Model"] != "Persistence_Baseline"]
    best_ml = (
        ml_results.sort_values(["MAE", "RMSE"]).iloc[0]
        if not ml_results.empty
        else None
    )
    best_overall = results.sort_values(["MAE", "RMSE"]).iloc[0]
    if best_ml is not None:
        st.success(
            f"Best ML model by held-out MAE for {horizon}: "
            f"**{best_ml['Model name']}** — MAE {best_ml['MAE']:.4f}, "
            f"RMSE {best_ml['RMSE']:.4f}, R² {best_ml['R2']:.4f}."
        )
    st.caption(
        f"Best overall, including baseline: {best_overall['Model name']}."
    )
    display = results[
        ["Model name", "MAE", "RMSE", "R2", "Test_Samples"]
    ].rename(columns={"R2": "R²", "Test_Samples": "Test samples"})
    st.dataframe(
        display.sort_values(["MAE", "RMSE"]),
        width="stretch",
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.4f"),
            "RMSE": st.column_config.NumberColumn(format="%.4f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "Test samples": st.column_config.NumberColumn(format="%d"),
        },
    )

    left, right = st.columns(2)
    with left:
        performance = go.Figure()
        for metric, color in (("MAE", COLORS["green"]), ("RMSE", COLORS["orange"])):
            performance.add_trace(
                go.Bar(
                    x=results["Model name"],
                    y=results[metric],
                    name=metric,
                    marker_color=color,
                )
            )
        performance.update_layout(
            barmode="group",
            xaxis_title="Model",
            yaxis_title="Error (moisture percentage points)",
        )
        st.plotly_chart(
            style_plot(performance, f"Model error comparison · {horizon}"),
            width="stretch",
            key="model_performance_errors",
        )
    with right:
        selected_model = st.selectbox(
            "Actual vs predicted model",
            list(MODEL_FILES),
            key="actual_predicted_model",
        )
        model_key = {
            "Linear Regression": "Linear_Regression",
            "Decision Tree": "Decision_Tree",
            "Random Forest": "Random_Forest",
        }[selected_model]
        actual = predictions.loc[
            (predictions["Forecast_Horizon"] == horizon)
            & (predictions["Model"] == model_key)
        ].copy()
        if actual.empty:
            st.info("No saved actual-versus-predicted rows match this selection.")
        else:
            actual["Timestamp"] = pd.to_datetime(
                actual["Target_Timestamp"], errors="coerce"
            )
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=actual["Actual_Moisture_Pct"],
                    y=actual["Predicted_Moisture_Pct"],
                    mode="markers",
                    name=MODEL_NAMES[model_key],
                    marker={"color": COLORS["green"], "opacity": 0.5, "size": 5},
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=actual["Actual_Moisture_Pct"],
                    y=actual["Persistence_Prediction_Pct"],
                    mode="markers",
                    name="Naive Persistence",
                    marker={"color": COLORS["orange"], "opacity": 0.35, "size": 4},
                )
            )
            lower = min(
                actual["Actual_Moisture_Pct"].min(),
                actual["Predicted_Moisture_Pct"].min(),
            )
            upper = max(
                actual["Actual_Moisture_Pct"].max(),
                actual["Predicted_Moisture_Pct"].max(),
            )
            figure.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    name="Perfect prediction",
                    line={"color": "#66756b", "dash": "dash"},
                )
            )
            figure.update_layout(
                xaxis_title="Actual moisture (%)",
                yaxis_title="Predicted moisture (%)",
            )
            st.plotly_chart(
                style_plot(figure, f"Actual vs predicted · {horizon}"),
                width="stretch",
                key="actual_predicted_scatter",
            )

    importance_model = st.selectbox(
        "Feature importance model",
        ["Random Forest", "Decision Tree"],
        key="importance_model",
    )
    artifact_path = MODELS_DIR / (
        f"{horizon}_{MODEL_FILES[importance_model]}.joblib"
    )
    try:
        artifact = load_saved_model(
            str(artifact_path), artifact_path.stat().st_mtime_ns
        )
        pipeline = artifact["pipeline"]
        estimator = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessing"]
        importances = getattr(estimator, "feature_importances_", None)
        feature_names = preprocessor.get_feature_names_out()
        if importances is None:
            st.info(f"{importance_model} does not expose feature importances.")
        elif len(importances) != len(feature_names):
            st.warning("Feature-importance values do not match transformed features.")
        else:
            importance_frame = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values("Importance", ascending=False).head(15)
            importance_frame["Feature"] = importance_frame["Feature"].str.replace(
                r"^(numeric__|farm__)", "", regex=True
            )
            importance_chart = px.bar(
                importance_frame.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[COLORS["green"]],
            )
            st.plotly_chart(
                style_plot(
                    importance_chart,
                    f"{importance_model} feature importance · {horizon}",
                    470,
                ),
                width="stretch",
                key="feature_importance",
            )
    except (OSError, KeyError, ValueError, AttributeError) as error:
        st.warning(f"Could not load feature importance for this model: {error}")


def irrigation_message(moisture, low_threshold, upper_threshold):
    if moisture < low_threshold:
        return (
            "Dry — irrigation may be required.",
            "The forecast is below the configured low-moisture threshold.",
        )
    if moisture > upper_threshold:
        return (
            "Wet — avoid irrigation.",
            "The forecast exceeds the configured upper saturation boundary.",
        )
    return (
        "Normal — monitor conditions.",
        "The forecast falls between the configured low and upper thresholds.",
    )


def show_irrigation_advisory(forecast_frame, historical, selected_farm, model_name):
    add_brand_header(
        "Irrigation Advisory System",
        "Use a saved soil-moisture forecast and adjustable demonstration thresholds.",
    )
    low_threshold = st.session_state["dry_threshold"]
    upper_threshold = st.session_state["upper_threshold"]
    if low_threshold >= upper_threshold:
        st.error(
            "The low-moisture threshold must be lower than the upper saturation boundary."
        )
        return
    horizon = st.selectbox(
        "Forecast horizon used for advisory",
        HORIZONS,
        key="irrigation_horizon",
    )
    try:
        latest = latest_forecast_row(forecast_frame, selected_farm)
        predictions = predict_horizons(
            latest, model_name, {}, load_saved_model
        )
    except (OSError, KeyError, ValueError, TypeError) as error:
        st.error(f"Could not generate an irrigation forecast: {error}")
        return
    forecast_value = predictions[horizon]["moisture"]
    farm_rows = historical.loc[historical[FARM_COLUMN] == selected_farm]
    if farm_rows.empty:
        st.error(f"No current historical observation is available for {selected_farm}.")
        return
    current_value = float(farm_rows.iloc[-1][MOISTURE_COLUMN])
    condition = condition_for(forecast_value, low_threshold, upper_threshold)
    action, reason = irrigation_message(
        forecast_value, low_threshold, upper_threshold
    )
    cards = st.columns(4)
    cards[0].metric("Current soil moisture", f"{current_value:.2f}%")
    cards[1].metric(f"{horizon} forecast", f"{forecast_value:.2f}%")
    cards[2].metric("Forecast soil condition", condition)
    cards[3].metric(
        "Recommended action",
        "Irrigation may be required"
        if forecast_value < low_threshold
        else "Avoid irrigation"
        if forecast_value > upper_threshold
        else "Monitor conditions",
    )
    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Recommendation")
        if forecast_value < low_threshold:
            st.warning(action)
        elif forecast_value > upper_threshold:
            st.error(action)
        else:
            st.success(action)
        st.write(f"**Reason:** {reason}")
        st.write(
            f"**Forecast horizon:** {horizon} "
            f"({predictions[horizon]['timestamp']:%Y-%m-%d %H:%M})"
        )
        st.write(f"**Farm:** {selected_farm} · **Model:** {model_name}")
        st.caption(
            f"Demonstration thresholds: dry < {low_threshold:.1f}%; "
            f"normal {low_threshold:.1f}%–{upper_threshold:.1f}%; "
            f"wet > {upper_threshold:.1f}%."
        )
    with right:
        st.plotly_chart(
            threshold_gauge(forecast_value, low_threshold, upper_threshold),
            width="stretch",
            key="irrigation_threshold_gauge",
        )
    st.warning(
        "These are demonstration thresholds, not universal agricultural standards. "
        "Calibrate them for crop type, soil texture, field capacity, irrigation "
        "method, and local agricultural conditions. Air humidity alone is not "
        "used to make this recommendation."
    )


def collector_state():
    """Initialize collector state for this browser session."""
    st.session_state.setdefault("collector_process", None)
    st.session_state.setdefault("collector_log_path", None)
    st.session_state.setdefault("collector_notice", "")


def _recent_collector_log():
    return collector_log_tail(st.session_state.get("collector_log_path"))


def _render_collector_controls():
    collector_state()
    process = st.session_state["collector_process"]
    running = process is not None and process.poll() is None
    status_text = "Collector running" if running else "Collector stopped"
    st.caption(
        f"Collector status: **{status_text}**. These buttons control only the "
        "Python serial collector; start/stop Wokwi separately in VS Code."
    )
    controls = st.columns(3)
    if controls[0].button(
        "▶ Start Reading", disabled=running, key="start_wokwi_collector"
    ):
        try:
            process, log_path = start_collector()
            st.session_state["collector_process"] = process
            st.session_state["collector_log_path"] = log_path
            time.sleep(0.25)
            if process.poll() is None:
                st.session_state["collector_notice"] = (
                    "Collector process started. It will save valid serial DATA "
                    "records when Wokwi's RFC2217 port is available."
                )
            else:
                st.session_state["collector_notice"] = (
                    "Collector exited during startup. Review its output below."
                )
        except OSError as error:
            st.session_state["collector_notice"] = (
                f"Could not start the collector: {error}"
            )
        st.rerun()

    if controls[1].button(
        "■ Stop Reading", disabled=not running, key="stop_wokwi_collector"
    ):
        try:
            stop_collector(process)
            st.session_state["collector_notice"] = (
                "Collector stopped. This does not stop Wokwi or change the ESP32."
            )
        except OSError as error:
            st.session_state["collector_notice"] = (
                f"Could not stop the collector cleanly: {error}"
            )
        st.rerun()

    if st.session_state["collector_notice"]:
        st.info(st.session_state["collector_notice"])
    if process is not None and process.poll() is not None:
        st.warning(
            f"Collector exited with code {process.returncode}. "
            "Confirm Wokwi is running and RFC2217 is listening on port 4000."
        )
    logs = _recent_collector_log()
    if logs:
        with st.expander("Collector output"):
            st.code("\n".join(logs), language="text")


def _prepare_hardware_data():
    if not HARDWARE_DATA_PATH.is_file():
        return pd.DataFrame()
    hardware = load_project_csv(
        str(HARDWARE_DATA_PATH), HARDWARE_DATA_PATH.stat().st_mtime_ns
    )
    if hardware.empty:
        return hardware
    required = {
        "Timestamp",
        "Soil_Moisture_Pct",
        "Temperature_C",
        "Air_Humidity_Pct",
        "Relay_Status",
    }
    missing = required - set(hardware.columns)
    if missing:
        raise ValueError(f"Wokwi CSV is missing columns: {sorted(missing)}")
    hardware["Timestamp_Display"] = hardware[TIMESTAMP_COLUMN].astype("string")
    hardware[TIMESTAMP_COLUMN] = pd.to_datetime(
        hardware[TIMESTAMP_COLUMN], errors="coerce", utc=True
    )
    for column in (
        "Soil_Moisture_Pct",
        "Temperature_C",
        "Air_Humidity_Pct",
        "Relay_Status",
    ):
        hardware[column] = pd.to_numeric(hardware[column], errors="coerce")
    return hardware.dropna(
        subset=[
            TIMESTAMP_COLUMN,
            "Soil_Moisture_Pct",
            "Temperature_C",
            "Air_Humidity_Pct",
            "Relay_Status",
        ]
    ).sort_values(TIMESTAMP_COLUMN)


def show_wokwi_demo():
    add_brand_header(
        "Wokwi Sensor Readings (Simulation)",
        "ESP32 simulator readings are separate from the historical training dataset.",
    )
    st.warning(
        "These are simulated Wokwi readings, not real agricultural measurements. "
        "This dashboard does not control Wokwi, LEDs, relays, or irrigation hardware."
    )
    hardware_path = HARDWARE_DATA_PATH
    if not hardware_path.is_file():
        st.info(
            "No hardware CSV exists yet. Start the Wokwi simulation, then select "
            "**Start Reading** to run the existing Python collector."
        )
        return
    if hardware_path.stat().st_size == 0:
        st.info("The hardware CSV is empty; no readings are displayed or fabricated.")
        return
    if st.button("🔄 Refresh Wokwi readings", key="refresh_wokwi_page"):
        load_project_csv.clear()
        st.rerun()
    try:
        hardware = _prepare_hardware_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not read the Wokwi CSV: {error}")
        return
    if hardware.empty:
        st.info("The hardware CSV contains no valid sensor records yet.")
        return

    latest = hardware.iloc[-1]
    moisture = float(latest["Soil_Moisture_Pct"])
    temperature = float(latest["Temperature_C"])
    humidity = float(latest["Air_Humidity_Pct"])
    relay_on = int(latest["Relay_Status"]) == 1
    mode = str(latest.get("Simulation_Mode", "Not recorded"))
    mode_is_simulated = (
        "simulat" in str(latest.get("Data_Source", "")).lower()
        or mode.upper() == "AUTO"
    )
    updated = str(latest["Timestamp_Display"]).replace("T", " ")
    st.subheader("Latest simulated sensor values")
    kpis = st.columns(4)
    kpis[0].metric("Soil Moisture", f"{moisture:.1f}%")
    kpis[1].metric("Temperature", f"{temperature:.1f} °C")
    kpis[2].metric("Air Humidity", f"{humidity:.1f}%")
    kpis[3].metric("Last Updated", updated)
    st.caption(
        f"Source: {latest.get('Data_Source', 'Not recorded')} · "
        f"Mode: {mode} · "
        f"{'SIMULATED' if mode_is_simulated else 'source as recorded'} · "
        f"{len(hardware):,} valid saved readings"
    )

    st.subheader("Sensor gauges and irrigation status")
    gauges = st.columns(4)
    gauges[0].plotly_chart(
        sensor_gauge(
            "Soil Moisture",
            moisture,
            1,
            100,
            "%",
            COLORS["green"],
            [
                {"range": [0, st.session_state["dry_threshold"]], "color": "#f8e9e9"},
                {
                    "range": [
                        st.session_state["dry_threshold"],
                        st.session_state["upper_threshold"],
                    ],
                    "color": "#eaf4ed",
                },
                {
                    "range": [st.session_state["upper_threshold"], 100],
                    "color": "#fff3d8",
                },
            ],
        ),
        width="stretch",
        key="wokwi_moisture_gauge",
    )
    gauges[1].plotly_chart(
        sensor_gauge(
            "Temperature",
            temperature,
            -40,
            80,
            " °C",
            COLORS["orange"],
            [{"range": [-40, 80], "color": "#f5f7f5"}],
        ),
        width="stretch",
        key="wokwi_temperature_gauge",
    )
    gauges[2].plotly_chart(
        sensor_gauge(
            "Air Humidity",
            humidity,
            1,
            100,
            "%",
            COLORS["blue"],
            [{"range": [0, 100], "color": "#edf5fa"}],
        ),
        width="stretch",
        key="wokwi_humidity_gauge",
    )
    with gauges[3]:
        st.toggle(
            "Irrigation Status",
            value=relay_on,
            disabled=True,
            help="Read-only status from the latest recorded Wokwi relay reading.",
            key="wokwi_irrigation_status",
        )
        st.caption("Recorded relay state (read-only)")

    st.subheader("Moisture status indicators")
    led_columns = st.columns(3)
    led_states = (
        (
            "Red LED · Dry",
            "#d64545",
            moisture < DRY_THRESHOLD,
            f"Soil moisture < {DRY_THRESHOLD}%",
            "Firmware turns the red LED on when irrigation is required.",
            "Physical LED",
        ),
        (
            "Green · Normal",
            "#2e9b62",
            DRY_THRESHOLD <= moisture <= WET_THRESHOLD,
            f"{DRY_THRESHOLD}% ≤ soil moisture ≤ {WET_THRESHOLD}%",
            "Dashboard-only normal-range indicator; no green LED is controlled by the firmware.",
            "Dashboard indicator",
        ),
        (
            "Yellow LED · Wet",
            "#e3b52f",
            moisture > WET_THRESHOLD,
            f"Soil moisture > {WET_THRESHOLD}%",
            "Firmware turns the yellow LED on when the soil is excessively wet.",
            "Physical LED",
        ),
    )
    for column, (label, color, is_on, range_text, description, indicator_type) in zip(
        led_columns, led_states
    ):
        with column:
            led_state = "ON" if is_on else "OFF"
            st.markdown(
                f"""
                <div style="border:1px solid #e2ebe4;border-radius:12px;
                            padding:14px;background:#fff;min-height:145px">
                  <div style="display:flex;align-items:center;gap:9px">
                    <span style="display:inline-block;width:15px;height:15px;
                                 border-radius:50%;background:{color};
                                 box-shadow:0 0 7px {color}"></span>
                    <strong>{label}</strong>
                  </div>
                  <div style="margin-top:10px;font-size:1.05rem">
                    Status: <strong>{led_state}</strong>
                  </div>
                  <div style="margin-top:5px;color:#526357">{range_text}</div>
                  <div style="margin-top:7px;color:#66756b;font-size:.86rem">
                    {description}
                  </div>
                  <div style="margin-top:7px;color:#66756b;font-size:.78rem">
                    {indicator_type}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.info(
        f"Current firmware moisture band: **{condition_for(moisture, DRY_THRESHOLD, WET_THRESHOLD)}**. "
        "The indicators above use the ESP32 firmware thresholds (30% and 70%). "
        "Only the red and yellow indicators represent physical LEDs; green is a "
        "dashboard-only indicator. The irrigation switch shows the recorded relay "
        "state and does not control Wokwi or hardware."
    )

    hardware["Soil_Condition"] = hardware["Soil_Moisture_Pct"].map(
        lambda value: condition_for(
            value,
            st.session_state["dry_threshold"],
            st.session_state["upper_threshold"],
        )
    )
    hardware["Relay_Display"] = hardware["Relay_Status"].map(
        {0: "OFF", 1: "ON"}
    ).fillna("Unknown")
    display_columns = [
        column
        for column in (
            "Timestamp_Display",
            "Soil_Moisture_Pct",
            "Temperature_C",
            "Air_Humidity_Pct",
            "Relay_Display",
            "Simulation_Mode",
            "Data_Source",
            "Soil_Condition",
        )
        if column in hardware.columns or column == "Relay_Display"
    ]
    recent = hardware.sort_values(
        TIMESTAMP_COLUMN, ascending=False
    ).loc[:, display_columns].head(20)
    recent = recent.rename(columns={"Timestamp_Display": TIMESTAMP_COLUMN})
    st.subheader("Recent Wokwi readings")
    st.dataframe(recent, width="stretch", hide_index=True)
    csv_bytes = hardware_path.read_bytes()
    st.download_button(
        "💾 Save to CSV",
        data=csv_bytes,
        file_name=hardware_path.name,
        mime="text/csv",
        help="Download a copy of the existing Wokwi CSV; no new records are created.",
        key="download_wokwi_csv",
    )


def main():
    try:
        historical_raw = load_required_frame(CLEANED_DATA_PATH)
        forecast_frame = load_required_frame(FORECAST_DATA_PATH)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load a required project dataset: {error}")
        st.info(
            "Run `python backend/clean_dataset.py` and "
            "`python backend/prepare_forecasting_data.py` from the project root."
        )
        st.stop()
    required = {
        TIMESTAMP_COLUMN,
        FARM_COLUMN,
        MOISTURE_COLUMN,
        "Soil_Temperature_C",
    }
    missing = required - set(historical_raw.columns)
    if missing:
        st.error(f"Cleaned historical data is missing columns: {sorted(missing)}")
        st.stop()
    historical = prepare_historical(historical_raw)
    farms = sorted(historical[FARM_COLUMN].dropna().astype(str).unique())
    if not farms:
        st.error("No farm IDs are available in the cleaned dataset.")
        st.stop()

    st.sidebar.image(str(LOGO_PATH), width=72)
    st.sidebar.markdown("## Precision Agriculture")
    st.sidebar.caption("Soil Moisture Forecasting")
    selected_page = st.sidebar.radio(
        "Navigation Menu",
        PAGE_OPTIONS,
        key="selected_dashboard_page",
    )
    st.sidebar.divider()
    st.sidebar.subheader("Forecast Settings")
    selected_farm = st.sidebar.selectbox(
        "Farm selection",
        farms,
        key="selected_farm",
    )
    selected_model = st.sidebar.selectbox(
        "Forecast model",
        list(MODEL_FILES),
        key="selected_model",
    )
    history_days = st.sidebar.slider(
        "Historical chart window (days)",
        min_value=1,
        max_value=30,
        value=7,
        key="history_window_days",
    )
    st.sidebar.subheader("Agronomic Water Thresholds")
    st.sidebar.caption("Demonstration thresholds; calibrate to field conditions.")
    low_threshold = st.sidebar.slider(
        "Low moisture threshold (%)",
        min_value=0,
        max_value=99,
        value=DRY_THRESHOLD,
        key="dry_threshold",
    )
    upper_threshold = st.sidebar.slider(
        "Upper saturation boundary (%)",
        min_value=1,
        max_value=100,
        value=WET_THRESHOLD,
        key="upper_threshold",
    )
    st.sidebar.info(
        "Below the low threshold: Dry. Between thresholds: Normal. "
        "Above the upper boundary: Wet. Thresholds vary by crop and soil."
    )

    if selected_page == PAGE_OPTIONS[0]:
        show_overview(historical, selected_farm)
    elif selected_page == PAGE_OPTIONS[1]:
        show_prediction(
            historical.loc[historical[FARM_COLUMN] == selected_farm].copy(),
            forecast_frame,
            selected_farm,
            selected_model,
            history_days,
        )
    elif selected_page == PAGE_OPTIONS[2]:
        show_visualization(historical)
    elif selected_page == PAGE_OPTIONS[3]:
        show_model_performance()
    elif selected_page == PAGE_OPTIONS[4]:
        show_irrigation_advisory(
            forecast_frame,
            historical,
            selected_farm,
            selected_model,
        )
    elif selected_page == PAGE_OPTIONS[5]:
        if IS_CLOUD_DEPLOYMENT:
            st.info(
                "Cloud mode displays the saved Wokwi readings snapshot. The serial "
                "collector cannot connect to a simulator running on your computer "
                "from this hosted dashboard."
            )
        else:
            _render_collector_controls()
        show_wokwi_demo()


if __name__ == "__main__":
    main()
