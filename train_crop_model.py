import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv("datasets/Crop_recommendation.csv")

# 2. Separate input features and output
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Crop Recommendation Model")
print("-------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)

# 8. Save the trained model
joblib.dump(model, "models/crop_model.pkl")

print("Model saved successfully!")