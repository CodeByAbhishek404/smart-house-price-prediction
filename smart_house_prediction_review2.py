# SmartHouse: Predictive Analytics for House Prices

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("C:/Users/HP/Desktop/Smarat House/Data/BostonHousing_sample.csv")

# Dataset size
print("Dataset size:", df.shape)

# Exploratory Data Analysis
print(df.describe())
print(df.info())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Distribution of house prices
sns.histplot(df['medv'], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price (in $1000s)")
plt.ylabel("Frequency")
plt.show()

# Feature and target selection
X = df.drop("medv", axis=1)
y = df["medv"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print train and test sizes
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Note: If test samples are less than 2, R2 score calculation will give a warning and 'nan' value.

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
print("Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest Regressor:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# Feature importance from Random Forest
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

