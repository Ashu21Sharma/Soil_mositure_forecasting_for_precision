# Soil Moisture Forecasting Dashboard Deployment

## Streamlit Community Cloud

1. Push this project to a **private GitHub repository**. Keep the processed CSV
   files, `reports/model_comparison.csv`, all nine files in `models/`, and
   `data/hardware/hardware_generated_dataset.csv` at their current paths.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from that repository and set the entry point to `dashboard/app.py`.
3. Use Python 3.12 in the app's advanced settings. The root `requirements.txt`
   pins the packages used to train and load the saved model artifacts.
4. To disable the local serial collector in the hosted dashboard, add this
   setting in the app's **Secrets** field:

   ```toml
   SOIL_MOISTURE_CLOUD_DEPLOYMENT = true
   ```

   The deployed Wokwi page will then display the committed readings snapshot.
   The serial collector cannot reach a simulator running on your own computer
   from the hosted server.
5. In app settings, set access to **Only specific people can view this app**.
   The private GitHub repository keeps the source and model/data files private;
   set app viewer access separately.

Do not commit `.venv/`, `.pio/`, machine-specific `.vscode/` files, or
credentials. The app requires these local project files:

- `data/processed/cleaned_soil_moisture.csv`
- `data/processed/forecasting_features.csv`
- `data/processed/test_predictions.csv`
- `reports/model_comparison.csv`
- The nine saved model artifacts in `models/`
- `data/hardware/hardware_generated_dataset.csv` (optional Wokwi snapshot)

Run locally from the project root with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```
