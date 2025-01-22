"""Main pipeline for house price prediction."""

import argparse
import yaml
import numpy as np
from src.data_loader import HousePriceDataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.models import HousePriceModels
from src.xgboost_model import XGBoostRegressor
from src.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--model", choices=["all", "lr", "rf", "gb", "xgb"], default="all")
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    loader = HousePriceDataLoader()
    loader.load(args.data)
    loader.describe()
    X_train, X_test, y_train, y_test = loader.split()

    engineer = FeatureEngineer()
    X_train = engineer.engineer(X_train)
    X_test = engineer.engineer(X_test)

    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    models = HousePriceModels()
    evaluator = ModelEvaluator()

    if args.model in ["all", "lr"]:
        models.train_linear_regression(X_train, y_train_log)
        preds = np.expm1(models.predict("linear_regression", X_test))
        evaluator.evaluate(y_test, preds, "Linear Regression")

    if args.model in ["all", "rf"]:
        cfg = config["models"]["random_forest"]
        models.train_random_forest(X_train, y_train_log, **cfg)
        preds = np.expm1(models.predict("random_forest", X_test))
        evaluator.evaluate(y_test, preds, "Random Forest")

    if args.model in ["all", "gb"]:
        cfg = config["models"]["gradient_boosting"]
        models.train_gradient_boosting(X_train, y_train_log, **cfg)
        preds = np.expm1(models.predict("gradient_boosting", X_test))
        evaluator.evaluate(y_test, preds, "Gradient Boosting")

    if args.model in ["all", "xgb"]:
        xgb_model = XGBoostRegressor(**config["models"]["xgboost"])
        xgb_model.train(X_train, y_train_log)
        preds = np.expm1(xgb_model.predict(X_test))
        evaluator.evaluate(y_test, preds, "XGBoost")

    print("\n" + "=" * 60)
    print(evaluator.comparison_table().to_string(index=False))


if __name__ == "__main__":
    main()
