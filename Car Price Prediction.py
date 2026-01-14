# =====================================
# Task 3 - Car Price Prediction
# =====================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv(r"C:\Users\anisc\Downloads\TASK 3\car data.csv")   

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

# -------------------------------------
# Feature Engineering
# -------------------------------------

# Extract brand from Car_Name
df["Brand"] = df["Car_Name"].apply(lambda x: x.split()[0])

# -------------------------------------
# Encode Categorical Columns
# -------------------------------------
le = LabelEncoder()

df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Selling_type"] = le.fit_transform(df["Selling_type"])
df["Transmission"] = le.fit_transform(df["Transmission"])
df["Brand"] = le.fit_transform(df["Brand"])

# -------------------------------------
# Feature Selection
# -------------------------------------
X = df[
    [
        "Year",
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Brand"
    ]
]

y = df["Selling_Price"]

# -------------------------------------
# Train-Test Split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------
# Linear Regression Model
# -------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n--- Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

# -------------------------------------
# Random Forest Model
# -------------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n--- Random Forest ---")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))

# -------------------------------------
# Visualization
# -------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Price")
plt.show()