import pandas as pd

data = pd.read_csv("datasets/daily_price.csv")

print(data.head())

print("\nDataset Shape:", data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())