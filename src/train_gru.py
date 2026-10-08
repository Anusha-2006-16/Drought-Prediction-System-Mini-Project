import pandas as pd
import  numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score,confusion_matrix )


df=pd.read_csv("../data/Drought_Prediction_Combined_Dataset.csv")

df["drought"]=(df["SPI12"]<-1).astype(int)

features=[
    "GWETTOP",
    "IMERG_PRECTOT",
    "RH2M",
    "T2M",
    "WS2M",
    "SPI1",
    "SPI3",
    "SPI6"
]

data=df[features].values
target=df["drought"].values

sequence_length=12

X=[]
y=[]

for i in range(sequence_length,len(df)):
    X.append(data[i-sequence_length:i])
    y.append(target[i])
X=np.array(X)
y=np.array(y)

split_index=int(len(X)*0.8)

X_train=X[:split_index]
X_test=X[split_index:]

y_train=y[:split_index]
y_test=y[split_index:]

print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# feature scaling

scaler=StandardScaler()
X_train_2d=X_train.reshape(-1,X_train.shape[2])
X_test_2d=X_test.reshape(-1,X_test.shape[2])

X_train_scaled=scaler.fit_transform(X_train_2d)
X_test_scaled=scaler.transform(X_test_2d)

X_train_scaled=X_train_scaled.reshape(X_train.shape)
X_test_scaled=X_test_scaled.reshape(X_test.shape)

# converting to pytorch tensors
X_train_tensor=torch.tensor(X_train_scaled,dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_scaled,dtype=torch.float32)

y_train_tensor=torch.tensor(y_train,dtype=torch.float32)
y_test_tensor=torch.tensor(y_test,dtype=torch.float32)

print("\nTensor shapes:")

print("X_train tensor:", X_train_tensor.shape)
print("X_test tensor:", X_test_tensor.shape)

print("y_train tensor:", y_train_tensor.shape)
print("y_test tensor:", y_test_tensor.shape)

class DroughtGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(64,1)
    def forward(self,x):
        out,_=self.gru(x)
        out=out[:,-1,:]
        out=self.dropout(out)
        out=self.fc(out)
        out=out.squeeze(1)
        return out
model=DroughtGRU()
print("\nGRU model:")
print(model)

positive_count=y_train_tensor.sum()
negative_count=len(y_train_tensor)-positive_count
pos_weight=negative_count/positive_count

print("\nPositive samples:",positive_count.item())
print("Negative samples:",negative_count.item())
print("Positive Weight:",pos_weight.item())

# Loss function
criterion=nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.reshape(1)
)
print("\nLoss Function: ")
print(criterion)

# Optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
print("\nOptimizer: ")
print(optimizer)

# Train GRU MODEL
epochs=100
for epoch in range(epochs):
    model.train()
    outputs=model(X_train_tensor)
    loss=criterion(outputs,y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch+1)%10 == 0:
        print(f"[{epoch+1}/{epochs}]: "
              f"Loss: {loss.item():.4f}")

# Model Evaluation
model.eval()
with torch.no_grad():
    test_outputs=model(X_test_tensor)
    test_probabilities=torch.sigmoid(test_outputs)
    test_predictions=(test_probabilities>=0.5).float()

y_true=y_test_tensor.numpy()
y_pred=test_predictions.numpy()
y_prob=test_probabilities.numpy()

# evaluation metrics

accuracy=accuracy_score(y_true, y_pred)
precision=precision_score(y_true, y_pred,zero_division=0)
recall=recall_score(y_true, y_pred,zero_division=0)
f1=f1_score(y_true, y_pred,zero_division=0)
cm=confusion_matrix(y_true, y_pred)

print("\n================================")
print("GRU EVALUATION METRICS")
print("================================")

print(f"Accuracy: {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"F1-Score: {f1:.4f}")


print("\nConfusion Matrix:")

print(cm)

print("\nRaw Logits / Probabilities:")

for i in range(len(y_true)):

    print(
        f"Sample {i + 1}: "
        f"Actual={int(y_true[i])}, "
        f"Logit={test_outputs[i].item():.6f}, "
        f"Probability={test_probabilities[i].item():.12f}, "
        f"Predicted={int(y_pred[i])}"
    )