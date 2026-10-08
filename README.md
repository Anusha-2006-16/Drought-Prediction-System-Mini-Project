# 🌾 Drought Prediction System Using Deep Learning

A Deep Learning-based web application that predicts drought conditions using historical environmental and precipitation-related data.

The system uses **LSTM, GRU, and 1D CNN** models to learn temporal patterns from the previous 12 months of environmental data. The final system uses an **LSTM model** for drought prediction and provides an interactive **Streamlit dashboard** for predictions.

---

## 📌 Project Overview

Drought is a major environmental problem that can affect agriculture, water availability, and ecosystems.

This project develops a Deep Learning-based drought prediction system using historical monthly environmental data.

The model analyzes:

- Soil moisture
- Precipitation
- Relative humidity
- Temperature
- Wind speed
- SPI1
- SPI3
- SPI6

The system uses the **previous 12 months of data** to predict whether the current/next period is likely to experience drought.

---

## 🎯 Objectives

- Analyze historical environmental data.
- Preprocess and normalize the input features.
- Create time-series sequences using 12 months of historical data.
- Train Deep Learning models for drought classification.
- Compare LSTM, GRU, and CNN models.
- Handle class imbalance during training.
- Save the trained model and scaler.
- Build an interactive Streamlit dashboard.
- Allow users to enter environmental parameters manually and obtain a drought prediction.

---

## 🧠 Deep Learning Models

Three Deep Learning architectures were implemented:

### 1. LSTM

Long Short-Term Memory networks are designed to learn patterns from sequential data.

Architecture:

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
