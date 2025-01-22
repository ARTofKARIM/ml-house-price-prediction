"""Regression models for house price prediction."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class HousePriceModels:
    """Collection of regression models for house price prediction."""

    def __init__(self):
        self.models = {}
        self.predictions = {}

    def train_linear_regression(self, X_train, y_train, regularization=None, alpha=1.0):
        if regularization == "ridge":
            model = Ridge(alpha=alpha)
        elif regularization == "lasso":
            model = Lasso(alpha=alpha)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        name = regularization or "linear_regression"
        self.models[name] = model
        train_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"{name} - Train RMSE: {rmse:.2f}")
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=200, max_depth=15, min_samples_split=5):
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models["random_forest"] = model
        train_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"Random Forest - Train RMSE: {rmse:.2f}")
        return model

    def train_gradient_boosting(self, X_train, y_train, n_estimators=300, learning_rate=0.05, max_depth=5):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=42,
        )
        model.fit(X_train, y_train)
        self.models["gradient_boosting"] = model
        train_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"Gradient Boosting - Train RMSE: {rmse:.2f}")
        return model

    def predict(self, model_name, X):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        preds = self.models[model_name].predict(X)
        self.predictions[model_name] = preds
        return preds

    def get_feature_importance(self, model_name):
        model = self.models[model_name]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            return np.abs(model.coef_)
        return None
