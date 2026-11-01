#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <LiquidCrystal_I2C.h>

#define SOIL_PIN 34
#define DHT_PIN 4
#define DHT_TYPE DHT22

#define RED_LED_PIN 18
#define RELAY_PIN 26
#define YELLOW_LED_PIN 19

#define LCD_ADDRESS 0x27
#define SDA_PIN 21
#define SCL_PIN 22

#define DRY_THRESHOLD 30
#define WET_THRESHOLD 70

#define SOIL_WET_RAW 0
#define SOIL_DRY_RAW 4095

#define RELAY_ACTIVE_HIGH true

enum class SimulationMode {
    AUTO,
    MANUAL
};

constexpr SimulationMode SIMULATION_MODE = SimulationMode::MANUAL;
constexpr unsigned long READING_INTERVAL_MS = 5000;

enum SoilCondition {
    SOIL_DRY,
    SOIL_NORMAL,
    SOIL_EXCESSIVELY_WET
};

DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(LCD_ADDRESS, 16, 2);

float simulateTriangle(
    uint32_t step,
    uint32_t halfPeriod,
    float minimum,
    float maximum,
    bool startsAtMaximum
) {
    const uint32_t cycleLength = halfPeriod * 2;
    const uint32_t phase = step % cycleLength;
    const uint32_t distance = phase <= halfPeriod ? phase : cycleLength - phase;
    const float progress = static_cast<float>(distance) / halfPeriod;
    const float range = maximum - minimum;

    return startsAtMaximum
        ? maximum - progress * range
        : minimum + progress * range;
}

void controlRelay(bool irrigationRequired) {
    if (RELAY_ACTIVE_HIGH) {
        digitalWrite(RELAY_PIN, irrigationRequired ? HIGH : LOW);
    } else {
        digitalWrite(RELAY_PIN, irrigationRequired ? LOW : HIGH);
    }
}

int readSoilMoisture() {
    long total = 0;

    for (int i = 0; i < 10; i++) {
        total += analogRead(SOIL_PIN);
        delay(10);
    }

    const int rawValue = total / 10;
    int moisture = map(rawValue, SOIL_WET_RAW, SOIL_DRY_RAW, 100, 0);
    moisture = constrain(moisture, 0, 100);

    Serial.print("Raw Sensor Value: ");
    Serial.println(rawValue);

    return moisture;
}

void displayReadings(
    int moisture,
    SoilCondition condition,
    float temperature,
    float humidity
) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Soil: ");
    lcd.print(moisture);
    lcd.print("% ");

    switch (condition) {
        case SOIL_DRY:
            lcd.print("DRY");
            break;
        case SOIL_NORMAL:
            lcd.print("OK");
            break;
        case SOIL_EXCESSIVELY_WET:
            lcd.print("WET");
            break;
    }

    lcd.setCursor(0, 1);
    if (isnan(temperature) || isnan(humidity)) {
        lcd.print("DHT22 read error");
    } else {
        lcd.print("T:");
        lcd.print(temperature, 1);
        lcd.print("C H:");
        lcd.print(humidity, 0);
        lcd.print("%");
    }
}

void setup() {
    Serial.begin(115200);

    pinMode(SOIL_PIN, INPUT);
    pinMode(RED_LED_PIN, OUTPUT);
    pinMode(YELLOW_LED_PIN, OUTPUT);
    pinMode(RELAY_PIN, OUTPUT);

    analogReadResolution(12);
    Wire.begin(SDA_PIN, SCL_PIN);

    lcd.init();
    lcd.backlight();
    dht.begin();

    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(YELLOW_LED_PIN, LOW);
    controlRelay(false);

    lcd.setCursor(0, 0);
    lcd.print("SOIL MOISTURE");
    lcd.setCursor(0, 1);
    lcd.print("FORECAST SYSTEM");

    Serial.println();
    Serial.println("================================");
    Serial.println("SOIL MOISTURE FORECASTING SYSTEM");
    Serial.println("PRECISION AGRICULTURE");
    Serial.println("================================");

    delay(2000);
    lcd.clear();
}

void loop() {
    static unsigned long lastReadingTime = 0;
    static uint32_t simulationStep = 0;
    const unsigned long now = millis();
    if (now - lastReadingTime < READING_INTERVAL_MS) {
        return;
    }
    lastReadingTime = now;

    Serial.println();
    Serial.println("------ SENSOR READINGS ------");

    int moisture;
    float temperature;
    float humidity;
    if (SIMULATION_MODE == SimulationMode::AUTO) {
        moisture = static_cast<int>(
            simulateTriangle(simulationStep, 35, 15.0f, 85.0f, true) + 0.5f
        );
        temperature = simulateTriangle(
            simulationStep, 24, 20.0f, 35.0f, false
        );
        humidity = simulateTriangle(
            simulationStep, 32, 35.0f, 85.0f, true
        );
        ++simulationStep;
    } else {
        moisture = readSoilMoisture();
        temperature = dht.readTemperature();
        humidity = dht.readHumidity();
    }

    Serial.print("Reading source: ");
    Serial.println(
        SIMULATION_MODE == SimulationMode::AUTO ? "AUTO" : "MANUAL"
    );

    Serial.print("Soil Moisture: ");
    Serial.print(moisture);
    Serial.println("%");

    if (isnan(temperature) || isnan(humidity)) {
        Serial.println("DHT22 reading failed!");
    } else {
        Serial.print("Temperature: ");
        Serial.print(temperature);
        Serial.println(" C");

        Serial.print("Humidity: ");
        Serial.print(humidity);
        Serial.println("%");
    }

    SoilCondition condition;
    if (moisture < DRY_THRESHOLD) {
        condition = SOIL_DRY;
    } else if (moisture > WET_THRESHOLD) {
        condition = SOIL_EXCESSIVELY_WET;
    } else {
        condition = SOIL_NORMAL;
    }

    const bool irrigationRequired = condition == SOIL_DRY;
    const char* conditionName =
        condition == SOIL_DRY ? "SOIL_DRY" :
        condition == SOIL_EXCESSIVELY_WET ? "SOIL_EXCESSIVELY_WET" :
        "SOIL_NORMAL";

    Serial.print("{\"moisture\":");
    Serial.print(moisture);
    Serial.print(",\"temperature\":");
    if (isnan(temperature)) {
        Serial.print("null");
    } else {
        Serial.print(temperature, 1);
    }
    Serial.print(",\"humidity\":");
    if (isnan(humidity)) {
        Serial.print("null");
    } else {
        Serial.print(humidity, 1);
    }
    Serial.print(",\"condition\":\"");
    Serial.print(conditionName);
    Serial.print("\",\"irrigationRequired\":");
    Serial.print(irrigationRequired ? "true" : "false");
    Serial.println("}");

    digitalWrite(RED_LED_PIN, irrigationRequired ? HIGH : LOW);
    digitalWrite(
        YELLOW_LED_PIN,
        condition == SOIL_EXCESSIVELY_WET ? HIGH : LOW
    );
    controlRelay(irrigationRequired);

    if (!isnan(temperature) && !isnan(humidity)) {
        const bool relayIsOn =
            digitalRead(RELAY_PIN) == (RELAY_ACTIVE_HIGH ? HIGH : LOW);
        Serial.print("MODE,");
        Serial.println(
            SIMULATION_MODE == SimulationMode::AUTO ? "AUTO" : "MANUAL"
        );
        Serial.print("DATA,");
        Serial.print(moisture);
        Serial.print(",");
        Serial.print(temperature, 2);
        Serial.print(",");
        Serial.print(humidity, 2);
        Serial.print(",");
        Serial.println(relayIsOn ? 1 : 0);
    }

    if (condition == SOIL_DRY) {
        Serial.println("STATUS: SOIL IS DRY");
        Serial.println("IRRIGATION: REQUIRED");
        Serial.println("RELAY: ON");
    } else if (condition == SOIL_EXCESSIVELY_WET) {
        Serial.println("STATUS: SOIL IS EXCESSIVELY WET");
        Serial.println("IRRIGATION: OFF");
        Serial.println("RELAY: OFF");
    } else {
        Serial.println("STATUS: SOIL MOISTURE NORMAL");
        Serial.println("IRRIGATION: NOT REQUIRED");
        Serial.println("RELAY: OFF");
    }

    displayReadings(moisture, condition, temperature, humidity);
    Serial.println("-----------------------------");
    Serial.println("-----------------------------");
}