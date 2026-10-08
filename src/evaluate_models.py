import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "../data/Drought_Prediction_Combined_Dataset.csv"
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

# Target
df["drought"] = (
        df["SPI12"] < -1
).astype(int)


# ============================================================
# 2. PREPARE DATA
# ============================================================

data = df[features].values
target = df["drought"].values

scaler = StandardScaler()

data = scaler.fit_transform(data)


# ============================================================
# 3. CREATE 12-MONTH SEQUENCES
# ============================================================

sequence_length = 12

X = []
y = []

for i in range(
        sequence_length,
        len(df)
):

    X.append(
        data[
            i-sequence_length:i
        ]
    )

    y.append(
        target[i]
    )


X = np.array(X)
y = np.array(y)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(X) * 0.8
)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# Convert to tensors

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# ============================================================
# 5. LSTM MODEL
# ============================================================

class DroughtLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.fc = nn.Linear(
            64,
            1
        )


    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc(out)

        out = out.squeeze(1)

        return out


# ============================================================
# 6. LOAD SAVED MODEL
# ============================================================

model = DroughtLSTM()

model.load_state_dict(
    torch.load(
        "../models/best_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ============================================================
# 7. PREDICTION
# ============================================================

with torch.no_grad():

    logits = model(
        X_test_tensor
    )

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
            probabilities >= 0.5
    ).int().numpy()


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print()
print("=" * 55)
print("FINAL SAVED LSTM MODEL EVALUATION")
print("=" * 55)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)

print("=" * 55)