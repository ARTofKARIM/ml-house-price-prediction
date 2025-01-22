"""Data loading and exploration for house price dataset."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


class HousePriceDataLoader:
    """Handles loading and initial analysis of housing data."""

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.data = None

    def load(self, filepath=None):
        path = filepath or self.config["data"]["path"]
        self.data = pd.read_csv(path)
        print(f"Dataset loaded: {self.data.shape}")
        return self.data

    def describe(self):
        if self.data is None:
            raise ValueError("Data not loaded")
        target = self.config["data"]["target"]
        summary = {
            "shape": self.data.shape,
            "numeric_features": len(self.data.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(self.data.select_dtypes(include=["object"]).columns),
            "missing_total": self.data.isnull().sum().sum(),
            "target_mean": self.data[target].mean(),
            "target_median": self.data[target].median(),
            "target_std": self.data[target].std(),
        }
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return summary

    def get_missing_info(self):
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        missing_pct = (missing / len(self.data) * 100).round(2)
        return pd.DataFrame({"count": missing, "percentage": missing_pct})

    def split(self):
        target = self.config["data"]["target"]
        X = self.data.drop(columns=[target])
        y = self.data[target]
        return train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
        )
