import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs, json
import tempfile
import requests
import base64
file_path = r"C:\Users\asus\Downloads\city_day.csv"
my_df = pd.read_csv(file_path)
print('The shape of our dataset is ', my_df.shape)
my_df.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Step 1: Data Acquisition
data = pd.read_csv( r"C:\Users\asus\Downloads\city_day.csv")

# Step 2: Data Cleaning
data = data.dropna()

# Step 3: Exploratory Data Analysis
print(data.describe())
data.hist(bins=20, figsize=(15,10))
plt.show()

# Step 4: Feature Identification
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'AQI'


# Step 6: Data Splitting
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Selection
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Step 8: Model Training and Evaluation
for name, model in models.items():
    print(f"Model: {name}")
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross Validation Scores: {scores}")
    print(f"Mean Cross Validation Score: {np.mean(scores)}")
    
    # Step 9: Model Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    
    # Step 10: Visualization of Predictions
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'{name} - Actual vs Predicted AQI')
    plt.show()
    
    # Step 11: Model Interpretation (not implemented here)

    # Step 12: Feature Importance Visualization (for Random Forest and Decision Tree)
    if name in ['Random Forest', 'Decision Tree']:
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
        plt.xticks(range(X.shape[1]), np.array(features)[sorted_indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature Importances ({name})')
        plt.show()
        
    if name in ['Random Forest', 'Decision Tree']:
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
        plt.xticks(range(X.shape[1]), np.array(features)[sorted_indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature Importances ({name})')
        plt.show()
    
    # Additional plot for Decision Tree
    if name == 'Decision Tree':
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI (Decision Tree)')
        plt.title('Decision Tree - Actual vs Predicted AQI')
        plt.show()

