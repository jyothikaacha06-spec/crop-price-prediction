import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("datasets/daily_price.csv")

# Inputs and target
X = data.drop("Modal Price", axis=1)
y = data["Modal Price"]

# Categorical columns
categorical_columns = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ],
    remainder="passthrough"
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Crop Price Prediction Model")
print("---------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(pipeline, "models/price_model.pkl")

print("Price model saved successfully!")