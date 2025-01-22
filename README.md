# House Price Prediction

A regression analysis project predicting residential property sale prices using multiple machine learning models including Linear Regression, Random Forest, Gradient Boosting, and XGBoost with comprehensive feature engineering.

## Overview

This project tackles the classic house price prediction problem using the Ames Housing dataset. It implements a full ML pipeline from exploratory analysis through feature engineering to model comparison.

## Architecture

```
ml-house-price-prediction/
├── src/
│   ├── data_loader.py          # Data ingestion and splitting
│   ├── eda.py                  # Exploratory data analysis
│   ├── preprocessing.py        # Missing values, encoding, scaling
│   ├── feature_engineering.py  # Derived features (area, age, quality)
│   ├── models.py               # Linear, Random Forest, Gradient Boosting
│   ├── xgboost_model.py        # XGBoost with hyperparameter tuning
│   ├── evaluation.py           # RMSE, MAE, R2, cross-validation
│   └── visualization.py        # Residual plots, feature importance, comparisons
├── config/config.yaml
├── tests/
└── main.py
```

## Models & Results

| Model | Approach | Key Hyperparameters |
|-------|----------|-------------------|
| Linear Regression | Baseline | fit_intercept=True |
| Random Forest | Ensemble (bagging) | n_estimators=200, max_depth=15 |
| Gradient Boosting | Ensemble (boosting) | n_estimators=300, lr=0.05 |
| XGBoost | Gradient boosting | n_estimators=500, lr=0.03 |

## Feature Engineering

- **Area features**: TotalArea, TotalLivingArea
- **Age features**: HouseAge, RemodAge, GarageAge
- **Quality features**: QualCondProduct, QualArea
- **Bathroom features**: TotalBathrooms
- **Log transformations** for skewed features

## Installation

```bash
git clone https://github.com/mouachiqab/ml-house-price-prediction.git
cd ml-house-price-prediction
pip install -r requirements.txt
```

## Usage

```bash
python main.py --data data/house_prices.csv --model all
python main.py --data data/house_prices.csv --model xgb
```

## Technologies

- Python 3.9+
- scikit-learn, XGBoost
- SHAP (model explainability)
- pandas, NumPy, Matplotlib, Seaborn






