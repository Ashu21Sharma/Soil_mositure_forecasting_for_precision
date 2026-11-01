#!/usr/bin/env python3
"""Append Wokwi ESP32 sensor readings from its RFC2217 serial port to CSV."""

import csv
import datetime
import math
import os
import sys
import tempfile
import time

import serial


SERIAL_URL = "rfc2217://localhost:4000"
BAUD_RATE = 115200
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "hardware", "hardware_generated_dataset.csv"
)
CSV_COLUMNS = (
    "Timestamp",
    "Soil_Moisture_Pct",
    "Temperature_C",
    "Air_Humidity_Pct",
    "Relay_Status",
    "Data_Source",
    "Simulation_Mode",
)


def parse_data_record(line):
    """Parse a DATA line, returning None for ordinary serial messages."""
    line = line.strip()
    if not line.startswith("DATA"):
        return None

    fields = line.split(",")
    if len(fields) != 5 or fields[0] != "DATA":
        raise ValueError("expected exactly five comma-separated DATA fields")

    try:
        soil_moisture = float(fields[1])
        temperature = float(fields[2])
        air_humidity = float(fields[3])
    except ValueError as error:
        raise ValueError("sensor readings must be numeric") from error

    if not all(
        math.isfinite(value)
        for value in (soil_moisture, temperature, air_humidity)
    ):
        raise ValueError("sensor readings must be finite numbers")
    if not 0 <= soil_moisture <= 100:
        raise ValueError("soil moisture must be between 0 and 100")
    if not 0 <= air_humidity <= 100:
        raise ValueError("air humidity must be between 0 and 100")
    if fields[4] not in ("0", "1"):
        raise ValueError("relay status must be 0 or 1")

    return soil_moisture, temperature, air_humidity, int(fields[4])


def parse_mode_record(line):
    """Return the simulation mode for a firmware MODE line."""
    fields = line.strip().split(",")
    if fields[0] != "MODE":
        return None
    if len(fields) != 2 or fields[1] not in ("AUTO", "MANUAL"):
        raise ValueError("expected MODE,AUTO or MODE,MANUAL")
    return fields[1]


def ensure_dataset_schema():
    """Create the CSV or migrate its header while preserving all existing rows."""
    output_directory = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_directory, exist_ok=True)

    if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0:
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(CSV_COLUMNS)
            csv_file.flush()
        return

    with open(OUTPUT_PATH, "r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        old_columns = reader.fieldnames or []
        if "Simulation_Mode" in old_columns:
            return
        old_rows = list(reader)

    output_columns = list(CSV_COLUMNS) + [
        column for column in old_columns if column not in CSV_COLUMNS
    ]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_directory,
            prefix=".hardware_generated_dataset.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            writer = csv.DictWriter(temporary_file, fieldnames=output_columns)
            writer.writeheader()
            for old_row in old_rows:
                row = {column: old_row.get(column, "") for column in output_columns}
                if not row["Simulation_Mode"]:
                    source = row.get("Data_Source", "")
                    row["Simulation_Mode"] = (
                        "AUTO"
                        if source == "Wokwi_Simulated_Data"
                        else "MANUAL"
                        if source == "Wokwi_Hardware_Data"
                        else ""
                    )
                writer.writerow(row)
            temporary_file.flush()

        os.replace(temporary_path, OUTPUT_PATH)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    saved_records = 0
    data_source = None
    warned_missing_mode = False
    try:
        ensure_dataset_schema()
        with serial.serial_for_url(
            SERIAL_URL, baudrate=BAUD_RATE, timeout=1
        ) as connection:
            with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

                print(f"Connected to Wokwi serial at {SERIAL_URL}.")
                print(f"Appending readings to {OUTPUT_PATH}. Press Ctrl+C to stop.")
                saw_serial_output = False
                next_status_at = time.monotonic() + 10

                while True:
                    raw_line = connection.readline()
                    if not raw_line:
                        now = time.monotonic()
                        if now >= next_status_at:
                            if saw_serial_output:
                                print(
                                    "Serial output is arriving, but no valid DATA "
                                    "records have arrived. The firmware sends DATA "
                                    "only when both DHT22 readings are valid; check "
                                    "for 'DHT22 reading failed!' in the Wokwi "
                                    "Serial Monitor."
                                )
                            else:
                                print(
                                    "No serial output received. Start the Wokwi "
                                    "simulator with 'Wokwi: Start Simulator' and "
                                    "confirm it is running this project's firmware."
                                )
                            next_status_at = now + 30
                        continue

                    saw_serial_output = True
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("MODE"):
                        try:
                            data_source = parse_mode_record(line)
                        except ValueError as error:
                            data_source = None
                            print(
                                f"Skipping invalid MODE record: {error}",
                                file=sys.stderr,
                            )
                        continue

                    try:
                        record = parse_data_record(line)
                    except ValueError as error:
                        print(f"Skipping invalid DATA record: {error}", file=sys.stderr)
                        continue

                    if record is None:
                        continue

                    if data_source is None:
                        if not warned_missing_mode:
                            print(
                                "Skipping DATA records until the firmware sends "
                                "MODE,SIMULATED or MODE,HARDWARE; this prevents "
                                "mixing unknown data sources.",
                                file=sys.stderr,
                            )
                            warned_missing_mode = True
                        continue

                    soil_moisture, temperature, air_humidity, relay_status = record
                    writer.writerow(
                        {
                            "Timestamp": datetime.datetime.now()
                            .astimezone()
                            .isoformat(timespec="seconds"),
                            "Soil_Moisture_Pct": soil_moisture,
                            "Temperature_C": temperature,
                            "Air_Humidity_Pct": air_humidity,
                            "Relay_Status": relay_status,
                            "Data_Source": (
                                "Wokwi_Simulated_Data"
                                if data_source == "AUTO"
                                else "Wokwi_Hardware_Data"
                            ),
                            "Simulation_Mode": data_source,
                        }
                    )
                    csv_file.flush()
                    saved_records += 1
                    print(f"Saved hardware reading {saved_records}.")
                    next_status_at = time.monotonic() + 30
                    warned_missing_mode = False
    except KeyboardInterrupt:
        print(f"\nStopped collection. Saved {saved_records} readings.")
        return 0
    except serial.SerialException as error:
        print(
            f"Could not connect to Wokwi serial at {SERIAL_URL}: {error}",
            file=sys.stderr,
        )
        return 1
    except OSError as error:
        print(
            f"Could not update or write the dataset at {OUTPUT_PATH}: {error}. "
            "Stop other collector instances, then run this collector again.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
