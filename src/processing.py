import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("../data/Drought_Prediction_Combined_Dataset.csv")

df["drought"]=(df["SPI12"]<=-1).astype(int)

features=[
    "GWETTOP",
    "IMERG_PRECTOT",
    "RH2M",
    "T2M",
    "WS2M",
    "SPI1",
    "SPI3",
    "SPI6",
    "SPI12"
]

data=df[features].values

target=df["drought"].values

sequence_length=12

X=[]
y=[]


for i in range(sequence_length, len(df)):
    X.append(data[i - sequence_length:i])
    y.append(target[i])

X=np.array(X)
y=np.array(y)

print("Original dataset shape:",df.shape)
print("\nSequence data shape:",X.shape)
print("Target shape:",y.shape)
print("\nFirst sequence shape:",X[0].shape)
print("\nFirst target:",y[0])

print("\nDrought distribution after sequence creation:")
print(pd.Series(y).value_counts())

split_index=int(len(X)*0.8)
X_train=X[:split_index]
X_test=X[split_index:]

y_train=y[:split_index]
y_test=y[split_index:]

print("\nTrain Test Split:")
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

print("\nTraining drought distribution:")
print(pd.Series(y_train).value_counts())
print("\nTesting drought distribution:")
print(pd.Series(y_test).value_counts())

scaler=StandardScaler()
X_train_2d=X_train.reshape(-1,X_train.shape[2])
X_test_2d=X_test.reshape(-1,X_test.shape[2])

scaler.fit(X_train_2d)

X_train_scaled=scaler.transform(X_train_2d)
X_test_scaled=scaler.transform(X_test_2d)

X_train_scaled=X_train_scaled.reshape(X_train.shape)
X_test_scaled=X_test_scaled.reshape(X_test.shape)

print("\nAfter scaling:")
print("X_train_scaled shape:",X_train_scaled.shape)
print("X_test_scaled shape:",X_test_scaled.shape)

print("\nSample scaled values:")
print(X_train_scaled[0])