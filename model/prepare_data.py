"""
prepare_data.py
---------------
Step 1 of the pipeline: fetch the Telco Customer Churn dataset, clean it and
carve out a stratified hold-out test set that is shipped with the repository.

Run from the repository root:
    python model/prepare_data.py

Outputs
    data/telco_churn_full.csv -> cleaned full dataset (7,032 x 20)
    data/train_data.csv       -> 80% stratified training split
    test_data.csv             -> 20% stratified hold-out split (uploaded in the app)
"""

import os
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

# IBM's public mirror of the Kaggle "Telco Customer Churn" dataset.
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")


def download_telco() -> pd.DataFrame:
    """Download the Telco Customer Churn CSV from the IBM public mirror."""
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
    print(f"Downloading dataset from {SOURCE_URL} ...")
    urlretrieve(SOURCE_URL, raw_path)
    return pd.read_csv(raw_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ID column, fix TotalCharges and remove rows with no charges."""
    print(f"Raw shape                 : {df.shape}")

    # customerID is a unique identifier, not a predictive feature.
    df = df.drop(columns=["customerID"])

    # TotalCharges arrives as text; 11 brand-new customers (tenure 0) hold a
    # blank string. Coerce to numeric and drop those rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = int(df["TotalCharges"].isna().sum())
    print(f"Blank TotalCharges rows   : {blanks}  (tenure-0 customers, dropped)")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # SeniorCitizen is coded 0/1; map it to No/Yes so every categorical
    # feature is human readable in the shipped CSV.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed    : {before - len(df)}")
    print(f"Cleaned shape             : {df.shape}")
    print(f"Features                  : {df.shape[1] - 1}")
    print(f"Churn distribution        :\n{df[TARGET].value_counts().to_string()}")
    return df


def main() -> None:
    df = clean(download_telco())
    df.to_csv(os.path.join(DATA_DIR, "telco_churn_full.csv"), index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    train_df.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    print(f"\nTrain split written       : data/train_data.csv  {train_df.shape}")
    print(f"Test  split written       : test_data.csv        {test_df.shape}")
    print("\nChurn distribution in the hold-out test set:")
    print(test_df[TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()
