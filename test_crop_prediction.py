import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/crop_model.pkl")

# Sample input values
input_data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 21,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 203
}])

# Predict the suitable crop
prediction = model.predict(input_data)

print("Crop Recommendation")
print("-------------------")
print("Recommended Crop:", prediction[0])