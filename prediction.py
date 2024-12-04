import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("heart_data.csv")

# Preprocess the data
X = data.drop(columns=["target"])
y = data["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

def predict_heart_disease(user_data):
    user_data = np.array(user_data).reshape(1, -1)
    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)
    return prediction[0]

def get_accuracy():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy * 100
