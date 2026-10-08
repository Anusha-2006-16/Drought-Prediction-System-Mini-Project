import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Drought Prediction System",
    page_icon="🌧️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Drought_Prediction_Combined_Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

df = pd.read_csv(
    DATA_PATH
)

features = [
    "GWETTOP",
    "IMERG_PRECTOT",
    "RH2M",
    "T2M",
    "WS2M",
    "SPI1",
    "SPI3",
    "SPI6"
]

class DroughtLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(64,1)
    def forward(self,x):
        out,_=self.lstm(x)
        out=out[:,-1,:]
        out=self.dropout(out)
        out=self.fc(out)
        out=out.squeeze(1)
        return out

model=DroughtLSTM()
model.load_state_dict(
    torch.load(
       MODEL_PATH,
        map_location=torch.device("cpu")
    )
)
model.eval()

scaler=joblib.load(
   SCALER_PATH
)

# ============================================================
# DASHBOARD
# ============================================================

st.title("🌧️ Drought Prediction System")

st.markdown(
    """
    ### Deep Learning Based Drought Prediction
    This system uses an **LSTM neural network** and the previous
    **12 months of environmental data** to predict the drought
    condition for the next month.
    """
)

st.divider()


# ============================================================
# LATEST DATA
# ============================================================

latest_date = df["time"].iloc[-1]

st.subheader("📅 Latest Available Data")

st.info(
    f"Latest dataset month: **{latest_date}**"
)


# ============================================================
# LATEST ENVIRONMENTAL VALUES
# ============================================================

st.subheader("🌍 Latest Environmental Conditions")

latest_row = df.iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🌱 Soil Moisture",
        f"{latest_row['GWETTOP']:.2f}"
    )

with col2:
    st.metric(
        "🌧️ Precipitation",
        f"{latest_row['IMERG_PRECTOT']:.2f}"
    )

with col3:
    st.metric(
        "💧 Humidity",
        f"{latest_row['RH2M']:.2f}"
    )

with col4:
    st.metric(
        "🌡️ Temperature",
        f"{latest_row['T2M']:.2f}"
    )

with col5:
    st.metric(
        "💨 Wind Speed",
        f"{latest_row['WS2M']:.2f}"
    )


st.divider()


# ============================================================
# SPI12 TREND
# ============================================================

st.subheader("📈 SPI12 Trend")

chart_data = df[["time", "SPI12"]].copy()

chart_data["time"] = pd.to_datetime(
    chart_data["time"]
)

chart_data = chart_data.set_index("time")

st.line_chart(
    chart_data,
    use_container_width=True
)

st.caption(
    "SPI12 is used to understand long-term precipitation-based "
    "drought conditions."
)


st.divider()


# ============================================================
# LATEST 12 MONTH DATA
# ============================================================

st.subheader("📊 Previous 12 Months")

last_12_months = df[features].tail(12)

st.dataframe(
    last_12_months,
    use_container_width=True
)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

st.subheader("🎯 Drought Prediction")

st.write(
    "Click the button below to predict the drought condition "
    "using the trained LSTM model."
)

if st.button(
        "🔮 Predict Drought",
        use_container_width=True
):

    # Get previous 12 months
    input_data = last_12_months.values

    # Scale data
    scaled_data = scaler.transform(
        input_data
    )

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        scaled_data,
        dtype=torch.float32
    )

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Model prediction
    with torch.no_grad():

        logit = model(input_tensor)

        probability = torch.sigmoid(
            logit
        ).item()

    probability_percentage = probability * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🔮 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Drought Probability",
            f"{probability_percentage:.2f}%"
        )

    with col2:

        if probability >= 0.5:

            st.error(
                "🔴 DROUGHT"
            )

        else:

            st.success(
                "🟢 NO DROUGHT"
            )


    # ========================================================
    # PROGRESS BAR
    # ========================================================

    st.write("Drought Probability")

    st.progress(
        probability
    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    if probability >= 0.5:

        st.warning(
            """
            The LSTM model predicts a **high probability of drought**
            for the next prediction period.
            """
        )

    else:

        st.success(
            """
            The LSTM model predicts a **low probability of drought**
            for the next prediction period.
            """
        )


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🤖 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Deep Learning Model",
        "LSTM"
    )

with col2:

    st.metric(
        "Sequence Length",
        "12 Months"
    )

with col3:

    st.metric(
        "Input Features",
        "8"
    )

with col4:

    st.metric(
        "Framework",
        "PyTorch"
    )


st.divider()


# ============================================================
# INPUT FEATURES
# ============================================================

st.subheader("📌 Features Used by the Model")

st.write(
    """
The LSTM model uses the following 8 features from the previous
12 months:

- 🌱 GWETTOP — Soil moisture
- 🌧️ IMERG_PRECTOT — Precipitation
- 💧 RH2M — Relative humidity
- 🌡️ T2M — Temperature
- 💨 WS2M — Wind speed
- 📉 SPI1 — 1-month Standardized Precipitation Index
- 📉 SPI3 — 3-month Standardized Precipitation Index
- 📉 SPI6 — 6-month Standardized Precipitation Index

**SPI12 is not directly given to the model as an input feature**
because the drought target is derived from SPI12.
"""
)


st.divider()


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.subheader("ℹ️ About the System")

st.write(
    """
This Drought Prediction System applies Deep Learning techniques
to historical environmental and precipitation-related data.

The system processes the previous 12 months of observations
and uses an LSTM network to identify patterns associated with
drought conditions.

The application provides:

✔ Historical environmental data

✔ SPI12 trend visualization

✔ Previous 12-month input data

✔ LSTM-based drought probability

✔ Drought / No Drought classification

✔ Model information
"""
)

# ============================================================
# MANUAL PARAMETER PREDICTION
# ============================================================

st.divider()

st.subheader("🧑‍💻 Manual Parameter Prediction")

st.write(
    """
Enter the environmental parameters for the latest month.
The system will combine your entered values with the previous
11 months of data and use the LSTM model to make the prediction.
"""
)

st.info(
    "💡 The LSTM requires 12 months of data. "
    "Your entered values represent the 12th/latest month."
)


# ============================================================
# INPUT FIELDS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    manual_gwettop = st.number_input(
        "🌱 GWETTOP (Soil Moisture)",
        value=float(df["GWETTOP"].iloc[-1]),
        step=0.01
    )

    manual_precipitation = st.number_input(
        "🌧️ IMERG_PRECTOT (Precipitation)",
        value=float(df["IMERG_PRECTOT"].iloc[-1]),
        step=0.01
    )

    manual_humidity = st.number_input(
        "💧 RH2M (Relative Humidity)",
        value=float(df["RH2M"].iloc[-1]),
        step=0.01
    )

    manual_temperature = st.number_input(
        "🌡️ T2M (Temperature)",
        value=float(df["T2M"].iloc[-1]),
        step=0.01
    )


with col2:

    manual_wind = st.number_input(
        "💨 WS2M (Wind Speed)",
        value=float(df["WS2M"].iloc[-1]),
        step=0.01
    )

    manual_spi1 = st.number_input(
        "📉 SPI1",
        value=float(df["SPI1"].iloc[-1]),
        step=0.01
    )

    manual_spi3 = st.number_input(
        "📉 SPI3",
        value=float(df["SPI3"].iloc[-1]),
        step=0.01
    )

    manual_spi6 = st.number_input(
        "📉 SPI6",
        value=float(df["SPI6"].iloc[-1]),
        step=0.01
    )


# ============================================================
# MANUAL PREDICTION BUTTON
# ============================================================

if st.button(
        "🔮 Predict Using Manual Parameters",
        use_container_width=True
):

    # Previous 11 months
    previous_11 = df[features].tail(11).values

    # User's manually entered month
    manual_row = np.array([
        manual_gwettop,
        manual_precipitation,
        manual_humidity,
        manual_temperature,
        manual_wind,
        manual_spi1,
        manual_spi3,
        manual_spi6
    ]).reshape(1, -1)


    # Combine previous 11 months + manual month
    manual_sequence = np.vstack([
        previous_11,
        manual_row
    ])


    # Scale using the SAME scaler used during training
    scaled_manual_sequence = scaler.transform(
        manual_sequence
    )


    # Convert to PyTorch tensor
    manual_tensor = torch.tensor(
        scaled_manual_sequence,
        dtype=torch.float32
    )


    # Add batch dimension
    manual_tensor = manual_tensor.unsqueeze(0)


    # Model prediction
    with torch.no_grad():

        manual_logit = model(
            manual_tensor
        )

        manual_probability = torch.sigmoid(
            manual_logit
        ).item()


    manual_probability_percentage = (
            manual_probability * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Manual Prediction Result"
    )

    st.metric(
        "Drought Probability",
        f"{manual_probability_percentage:.2f}%"
    )

    st.progress(
        manual_probability
    )


    if manual_probability >= 0.5:

        st.error(
            "🔴 DROUGHT"
        )

        st.write(
            "Based on the entered parameters, "
            "the LSTM model predicts a high probability "
            "of drought."
        )

    else:

        st.success(
            "🟢 NO DROUGHT"
        )

        st.write(
            "Based on the entered parameters, "
            "the LSTM model predicts a low probability "
            "of drought."
        )