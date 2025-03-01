import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/customer_churn_telecom_services.csv")
    return df

def preprocess_data(df):
    df = df.dropna()
    df.to_csv("data/processed/cleaned_data.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    preprocess_data(df)
    print("Data processing completed.")
