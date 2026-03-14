# Credit Card Fraud Detection

## Overview
An end-to-end machine learning pipeline built to detect fraudulent credit card 
transactions. This project addresses the challenge of severe class imbalance 
(only ~0.2% of transactions are fraudulent) using SMOTE oversampling and 
evaluates model performance using F1 Score and ROC-AUC rather than standard 
accuracy.

## Dataset
- **Source:** ULB Credit Card Fraud Detection Dataset
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions
- **Features:** V1–V28 (PCA-transformed), Time, Amount
- **Target:** Class (0 = Legitimate, 1 = Fraudulent)

> ⚠️ Dataset not included in this repo due to size. Download directly from Kaggle.

## Pipeline Steps
1. Exploratory Data Analysis
2. Feature Engineering
3. Data Splitting & Scaling
4. Handling Class Imbalance (SMOTE)
5. Baseline Model - Logistic Regression
6. Main Model - XGBoost
7. Model Evaluation & Comparison
8. Model Explainability - SHAP

## Tools & Libraries
- Python 3.10
- Pandas & NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- SHAP
- Matplotlib & Seaborn

## How to Run
1. Clone this repository
2. Download the dataset from Kaggle and place `creditcard.csv` in the project folder
3. Open `Fraud_Detection_Clean.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells in order

## Author
Uthman Olapade -Data Science & AI Student
