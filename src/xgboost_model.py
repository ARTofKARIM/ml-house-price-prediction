"""XGBoost regression model with hyperparameter tuning."""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


class XGBoostRegressor:
    """XGBoost wrapper for house price prediction."""

    def __init__(self, n_estimators=500, learning_rate=0.03, max_depth=6):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        self.model = xgb.XGBRegressor(**self.params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
        )
        train_pred = self.model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"XGBoost - Train RMSE: {rmse:.2f}")
        return self.model

    def hyperparameter_search(self, X_train, y_train, cv=5):
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.1],
            "n_estimators": [300, 500],
            "subsample": [0.7, 0.8],
        }
        base_model = xgb.XGBRegressor(random_state=42)
        grid = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1,
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        self.model = grid.best_estimator_
        print(f"Best params: {self.best_params}")
        print(f"Best CV RMSE: {-grid.best_score_:.2f}")
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self, importance_type="gain"):
        return self.model.feature_importances_

    def save(self, path="models/xgboost_model.json"):
        self.model.save_model(path)

    def load(self, path="models/xgboost_model.json"):
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
