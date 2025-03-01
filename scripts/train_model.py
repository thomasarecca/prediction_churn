import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load processed data
df = pd.read_csv("data/processed/cleaned_data.csv")

# Encode categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns
if len(categorical_cols) > 0:
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders in case they are needed later

# Split into features (X) and target (y)
X = df.drop(columns=['Churn'])  # Change 'Churn' to your actual target column
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model training completed and saved as model.pkl")
