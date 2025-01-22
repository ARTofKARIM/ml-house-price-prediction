"""Model evaluation with cross-validation and residual analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


class ModelEvaluator:
    """Evaluates regression models with multiple metrics."""

    def __init__(self):
        self.results = {}

    def evaluate(self, y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        self.results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
        print(f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%")
        return self.results[model_name]

    def cross_validate(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        rmse_scores = -scores
        print(f"CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
        return rmse_scores

    def residual_analysis(self, y_true, y_pred):
        residuals = y_true - y_pred
        return {
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "max_overpredict": np.min(residuals),
            "max_underpredict": np.max(residuals),
        }

    def comparison_table(self):
        rows = []
        for name, m in self.results.items():
            rows.append({"Model": name, **{k: f"{v:.2f}" for k, v in m.items()}})
        return pd.DataFrame(rows)
