"""Exploratory Data Analysis for house price data."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryAnalysis:
    """Generates EDA visualizations and statistics."""

    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def correlation_matrix(self, df, target, top_n=15, save=True):
        """Plot correlation matrix for top correlated features."""
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
        top_features = correlations.head(top_n).index

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df[top_features].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax, center=0)
        ax.set_title(f"Top {top_n} Feature Correlations")
        if save:
            fig.savefig(f"{self.output_dir}correlation_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return correlations

    def target_distribution(self, y, save=True):
        """Plot target variable distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(y, bins=50, color="steelblue", edgecolor="black")
        ax1.set_title("Sale Price Distribution")
        ax1.set_xlabel("Price")

        ax2.hist(np.log1p(y), bins=50, color="coral", edgecolor="black")
        ax2.set_title("Log Sale Price Distribution")
        ax2.set_xlabel("Log(Price)")
        if save:
            fig.savefig(f"{self.output_dir}target_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def scatter_top_features(self, df, target, top_n=6, save=True):
        """Scatter plots of top correlated features vs target."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
        features = [f for f in corr.index if f != target][:top_n]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for idx, feat in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            ax.scatter(df[feat], df[target], alpha=0.3, s=10, color="steelblue")
            ax.set_xlabel(feat)
            ax.set_ylabel(target)
            ax.set_title(f"{feat} vs {target}")
        plt.tight_layout()
        if save:
            fig.savefig(f"{self.output_dir}scatter_features.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
