import pandas as pd


DATA_PATH = "../data/Telco-Churn.csv"

TARGET = "Churn"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nChurn distribution:")
print(df[TARGET].value_counts(dropna=False))
print("\nChurn proportions:")
print(df[TARGET].value_counts(normalize=True).round(3))

# TotalCharges looks numeric but may contain blank strings in the original CSV.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# CustomerID is an identifier, not a behavioural predictor.
df = df.drop(columns=["customerID"], errors="ignore")

# Encode the target: No = 0, Yes = 1.
df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

if df[TARGET].isna().any():
    raise ValueError("Unexpected target labels. Expected only 'Yes' and 'No'.")

X = df.drop(columns=TARGET)
y = df[TARGET].astype(int)

# print("Feature matrix:", X.shape)
# print("Target vector:", y.shape)
# print("\nMissing values by feature (non-zero only):")
# print(X.isna().sum()[X.isna().sum() > 0])