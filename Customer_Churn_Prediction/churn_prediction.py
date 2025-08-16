import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Download the Telco Customer Churn dataset from a public source
DATA_URL = 'https://github.com/IBM/telco-customer-churn-on-icp4d/raw/master/data/Telco-Customer-Churn.csv'
DATA_PATH = 'Telco-Customer-Churn.csv'

if not os.path.exists(DATA_PATH):
    print('Downloading dataset...')
    response = requests.get(DATA_URL)
    with open(DATA_PATH, 'wb') as f:
        f.write(response.content)
    print('Download complete.')
else:
    print('Dataset already exists.')


# Load dataset with error handling
print('Loading dataset...')
try:
    df = pd.read_csv(DATA_PATH)
    if df.shape[1] == 1:
        print('Warning: Only one column detected. Trying with delimiter ",".')
        df = pd.read_csv(DATA_PATH, delimiter=',')
    if df.empty:
        raise ValueError('Dataset is empty. Please check the download or source URL.')
    print('Dataset loaded. Shape:', df.shape)
    print('Columns:', df.columns.tolist())
except Exception as e:
    print('Error loading dataset:', e)
    exit(1)

# Data preprocessing
print('Preprocessing data...')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    if col != 'customerID':
        df[col] = LabelEncoder().fit_transform(df[col])

# Feature selection
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
print('Training model...')
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print('Model training complete.')

# Prediction and evaluation
print('Evaluating model...')
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Predict for a random sample
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print('Sample prediction (0=No Churn, 1=Churn):', prediction[0])

# --- Visualizations ---

plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Churn distribution

plt.figure(figsize=(5,3))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')
plt.close()

# Feature importance (coefficients)

feature_importance = pd.Series(model.coef_[0], index=X.columns)
plt.figure(figsize=(7,5))
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Coefficient Value')
plt.savefig('feature_importance.png')
plt.close()

# --- Brief Report / Summary ---
print("\n--- Project Summary ---")
print(f"Model: Logistic Regression")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Top 5 Features Contributing to Churn:")
print(feature_importance.nlargest(5))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
