# 🌾 Drought Prediction System Using Deep Learning

A Deep Learning-based web application that predicts drought conditions using historical environmental and precipitation-related data.

The system implements **LSTM, GRU, and 1D CNN** models to learn temporal patterns from 12 months of historical environmental data. The final system uses an **LSTM model** for drought prediction and provides an interactive **Streamlit dashboard** for prediction and manual parameter input.

---

## 📌 Project Overview

Drought is a major environmental problem that can affect agriculture, water availability, and ecosystems.

This project develops a Deep Learning-based drought prediction system using historical monthly environmental data.

The system analyzes the following environmental and precipitation-related parameters:

- 🌱 Soil Moisture
- 🌧️ Precipitation
- 💧 Relative Humidity
- 🌡️ Temperature
- 💨 Wind Speed
- 📉 SPI1
- 📉 SPI3
- 📉 SPI6

The model uses the **previous 12 months of data** to learn temporal patterns and predict whether the target period is likely to experience drought.

---

## 🎯 Objectives

- Analyze historical environmental data.
- Preprocess and normalize input features.
- Create time-series sequences using 12 months of historical data.
- Train Deep Learning models for drought classification.
- Compare LSTM, GRU, and 1D CNN models.
- Handle class imbalance during model training.
- Save the trained model and feature scaler.
- Build an interactive Streamlit dashboard.
- Provide automatic drought prediction.
- Allow users to enter environmental parameters manually and obtain a drought prediction.

---

## 🧠 Deep Learning Models

Three Deep Learning architectures were implemented.

### 1. LSTM

Long Short-Term Memory networks are designed to learn patterns from sequential data.

### Architecture

```text
Input
  ↓
LSTM
  ↓
Dropout
  ↓
Fully Connected Layer
  ↓
Drought Probability


### 🔴 IMPORTANT — your 6 images must have these exact names

In the **same folder as `README.md`** on GitHub:


dashboard.png
spi12-trend.png
prediction.png
manual-input.png
manual-result.png
output.png
