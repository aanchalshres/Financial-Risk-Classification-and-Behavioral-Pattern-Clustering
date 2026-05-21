# Financial-Risk-Classification-and-Behavioral-Pattern-Clustering

Financial risk classification and customer behavioral clustering on loan approval datasets.

## Overview

This project applies machine learning to classify loan approval risk and segment customers into behavioral groups based on financial profiles. The analysis includes data preprocessing, exploratory analysis, unsupervised clustering, and supervised classification with strong performance metrics.

## Key Features

- **Risk Classification**: Predicts loan approval likelihood with 86.7% accuracy and 92.6% ROC-AUC
- **Customer Segmentation**: Identifies 3 distinct customer clusters based on demographics and financial behavior
- **Feature Engineering**: Processes 12+ features including income, credit score, debt-to-income ratio, employment status
- **Balanced Dataset**: Maintains 76.1% / 23.9% class distribution in stratified samples
- **Scalable Pipeline**: Trained on 20,000 records with preprocessing and model persistence

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── raw_data.csv (20,000 rows)
│   └── processed/
│       ├── clustered_data.csv
│       ├── preprocessed_sample.csv
│       ├── X_train.csv, X_test.csv, X_scaled.csv
│       └── y_train.csv, y_test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_classification.ipynb
├── models/
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
└── plots/
    ├── eda/
    ├── preprocessing/
    ├── clustering/
    └── classification/
```

## Approach

- **Classification**: Random Forest 
- **Clustering**: KMeans 
- **Preprocessing**: Feature scaling, categorical encoding, train-test split (80/20)

## Technologies

- **Python 3.x**
- **pandas** - Data manipulation
- **scikit-learn** - ML models and preprocessing
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Visualization
- **jupyter** - Interactive notebooks

## Data Source

Kaggle Link - https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval (20,000 records with 12+ features)