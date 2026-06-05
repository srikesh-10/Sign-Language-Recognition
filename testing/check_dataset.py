import pandas as pd

df = pd.read_csv("dataset/raw/sign_landmarks.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())