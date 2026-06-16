import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("telco_customer_churn.csv")
print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].median()
)

df.drop('customerID', axis=1, inplace=True)

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state=42
)

print(df.isnull().sum())

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



