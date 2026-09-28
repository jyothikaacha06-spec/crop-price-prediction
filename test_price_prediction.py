import pandas as pd
import joblib

# Load the trained price model
model = joblib.load("models/price_model.pkl")

# Sample input
input_data = pd.DataFrame([{
    "State": "Andhra Pradesh",
    "District": "Chittor",
    "Market": "Chittoor",
    "Commodity": "Rice",
    "Variety": "Common",
    "Grade": "FAQ",
    "Min Price": 3200,
    "Max Price": 3400
}])

# Predict modal price
prediction = model.predict(input_data)

print("Crop Price Prediction")
print("---------------------")
print("Predicted Modal Price: ₹", round(prediction[0], 2))