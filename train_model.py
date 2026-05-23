import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("heart_disease_dataset_5000.csv")

# Features
X = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]

# Target
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy * 100)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")