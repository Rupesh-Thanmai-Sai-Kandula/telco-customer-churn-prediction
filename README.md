# Telco Customer Churn Prediction

## Project Overview

Customer churn is one of the most important business challenges in the telecommunications industry. Acquiring a new customer is significantly more expensive than retaining an existing one. This project focuses on identifying customers who are likely to discontinue telecom services using machine learning techniques.

The objective is to build a predictive model that can classify whether a customer is likely to churn based on demographic information, service subscriptions, account details, and billing characteristics.


## Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains customer-level information including:

* Demographics

  * Gender
  * Senior Citizen Status
  * Partner
  * Dependents

* Account Information

  * Tenure
  * Contract Type
  * Payment Method
  * Paperless Billing

* Services

  * Phone Service
  * Multiple Lines
  * Internet Service
  * Online Security
  * Online Backup
  * Device Protection
  * Tech Support
  * Streaming TV
  * Streaming Movies

* Billing Information

  * Monthly Charges
  * Total Charges

* Target Variable

  * Churn (Yes/No)


## Project Workflow

### 1. Data Cleaning

* Removed duplicate records
* Converted `TotalCharges` from object datatype to numeric datatype
* Handled invalid values using `errors='coerce'`
* Imputed missing values in `TotalCharges` using the median
* Removed the non-informative `customerID` column

### 2. Data Preprocessing

* Applied Label Encoding to categorical variables
* Split data into features (`X`) and target (`y`)
* Performed Train-Test Split using an 80:20 ratio

### 3. Machine Learning Model

Algorithm Used:

* Logistic Regression

Training Configuration:

* Train/Test Split: 80/20
* Random State: 42

### 4. Model Evaluation

Evaluation Metrics:

* Accuracy Score
* Confusion Matrix
* Classification Report

Current Baseline Results:

* Accuracy: 81.83%

Confusion Matrix:

|                 | Predicted No Churn | Predicted Churn |
| --------------- | ------------------ | --------------- |
| Actual No Churn | 934                | 102             |
| Actual Churn    | 154                | 219             |

Classification Performance:

| Metric    | No Churn | Churn |
| --------- | -------- | ----- |
| Precision | 0.86     | 0.68  |
| Recall    | 0.90     | 0.59  |
| F1-Score  | 0.88     | 0.63  |


## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn


## Repository Structure

```text
Telco-Customer-Churn/
│
├── data/
│   └── telco_customer_churn.csv
│
├── src/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```


## Future Enhancements

### Exploratory Data Analysis (EDA)

* Churn by Contract Type
* Churn by Internet Service
* Churn by Payment Method
* Monthly Charges Analysis
* Tenure Analysis

### Model Improvements

* Feature Scaling
* Random Forest Classifier
* XGBoost
* Hyperparameter Tuning
* Cross Validation

### Business Intelligence

* Power BI Dashboard
* Customer Segmentation
* Churn Risk Monitoring Dashboard


## Author

Rupesh Thanmai Sai Kandula

This project was developed as part of a machine learning and business analytics learning journey focused on customer churn prediction and data-driven decision making.
