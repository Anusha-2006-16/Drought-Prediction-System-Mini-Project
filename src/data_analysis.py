import pandas as pd
df=pd.read_csv("../data/Drought_Prediction_Combined_Dataset.csv")

print("Dataset shape:",df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())