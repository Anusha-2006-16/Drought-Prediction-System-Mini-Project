import pandas as pd
import torch
import torch.nn as nn
import joblib

df = pd.read_csv("../data/Drought_Prediction_Combined_Dataset.csv")

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
        "../models/best_model.pth",
        map_location=torch.device("cpu")
    )
)
model.eval()

scaler=joblib.load("../models/scaler.pkl")

last_12_months=df[features].tail(12).values

print("Input shape:",last_12_months.shape)

scaled_data=scaler.transform(last_12_months)

input_tensor=torch.tensor(
    scaled_data,dtype=torch.float32
)
input_tensor=input_tensor.unsqueeze(0)

print("Tensor shape:",input_tensor.shape)

with torch.no_grad():
    logit=model(input_tensor)
    probability=torch.sigmoid(logit).item()

print("\n"+"="*50)
print("DROUGHT PREDICTION:")
print("="*50)

print(f"Drought Probability: {probability*100:.2f}%")

if probability>=0.5:
    print("Prediction: DROUGHT")
else:
    print("Prediction: NO DROUGHT")
print("="*50)