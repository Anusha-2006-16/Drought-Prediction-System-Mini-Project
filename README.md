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

###OUTPUT
<img width="959" height="478" alt="image" src="https://github.com/user-attachments/assets/7b7f86e1-5d11-476f-aaf7-1bd89e929ebd" />
<img width="957" height="475" alt="image" src="https://github.com/user-attachments/assets/6b3de8df-6b01-486f-a7b8-b7bae4e710f4" />
<img width="959" height="478" alt="image" src="https://github.com/user-attachments/assets/7ef89cc7-25fb-418c-9fce-a4f025b878e3" />
<img width="959" height="473" alt="image" src="https://github.com/user-attachments/assets/d4aade36-25c1-4445-927a-a2a75c5ae2bd" />
<img width="815" height="408" alt="image" src="https://github.com/user-attachments/assets/0559347d-8c83-460e-84b7-f9eacd80c820" />
<img width="842" height="419" alt="image" src="https://github.com/user-attachments/assets/d21a6f8e-a4f9-46e4-8b69-de1870d9cdb6" />
<img width="912" height="470" alt="image" src="https://github.com/user-attachments/assets/435c9440-8302-4838-bbfb-5a8dfef4c5a0" />
<img width="959" height="419" alt="image" src="https://github.com/user-attachments/assets/72b3877f-137d-46fe-9d84-3e51cdf33bbe" />








