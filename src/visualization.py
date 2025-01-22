"""Visualization suite for house price prediction results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class PriceVisualizer:
    """Generates regression analysis plots."""

    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, save=True):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2)
        ax.set_xlabel("Actual Price", fontsize=12)
        ax.set_ylabel("Predicted Price", fontsize=12)
        ax.set_title(f"Predicted vs Actual - {model_name}", fontsize=14)
        if save:
            fig.savefig(f"{self.output_dir}pred_vs_actual_{model_name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_residuals(self, y_true, y_pred, model_name, save=True):
        residuals = y_true - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.scatter(y_pred, residuals, alpha=0.4, s=10, color="steelblue")
        ax1.axhline(y=0, color="red", linestyle="--")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Residual")
        ax1.set_title(f"Residual Plot - {model_name}")

        ax2.hist(residuals, bins=50, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Residual")
        ax2.set_title("Residual Distribution")
        if save:
            fig.savefig(f"{self.output_dir}residuals_{model_name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_feature_importance(self, importances, feature_names, model_name, top_n=15, save=True):
        indices = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_n), importances[indices], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f"Feature Importance - {model_name}")
        if save:
            fig.savefig(f"{self.output_dir}importance_{model_name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_learning_curve(self, train_sizes, train_scores, val_scores, model_name, save=True):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, -train_scores.mean(axis=1), label="Train RMSE", color="steelblue")
        ax.fill_between(train_sizes, -train_scores.mean(axis=1) - train_scores.std(axis=1),
                        -train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        ax.plot(train_sizes, -val_scores.mean(axis=1), label="Val RMSE", color="coral")
        ax.fill_between(train_sizes, -val_scores.mean(axis=1) - val_scores.std(axis=1),
                        -val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        ax.set_xlabel("Training Size")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Learning Curve - {model_name}")
        ax.legend()
        if save:
            fig.savefig(f"{self.output_dir}learning_curve_{model_name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_model_comparison(self, results_dict, metric="RMSE", save=True):
        models = list(results_dict.keys())
        values = [results_dict[m][metric] for m in models]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, values, color="steelblue")
        ax.set_ylabel(metric)
        ax.set_title(f"Model Comparison - {metric}")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=15)
        if save:
            fig.savefig(f"{self.output_dir}model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
