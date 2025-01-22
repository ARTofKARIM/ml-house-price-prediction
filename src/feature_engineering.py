"""Feature engineering for house price prediction."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Creates derived features from housing data."""

    def __init__(self):
        self.new_features = []

    def create_area_features(self, df):
        df = df.copy()
        if "TotalBsmtSF" in df.columns and "1stFlrSF" in df.columns:
            df["TotalArea"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
            self.new_features.append("TotalArea")
        if "GrLivArea" in df.columns and "TotalBsmtSF" in df.columns:
            df["TotalLivingArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
            self.new_features.append("TotalLivingArea")
        return df

    def create_age_features(self, df):
        df = df.copy()
        if "YearBuilt" in df.columns and "YrSold" in df.columns:
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
            df["HouseAge"] = df["HouseAge"].clip(lower=0)
            self.new_features.append("HouseAge")
        if "YearRemodAdd" in df.columns and "YrSold" in df.columns:
            df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
            df["RemodAge"] = df["RemodAge"].clip(lower=0)
            self.new_features.append("RemodAge")
        if "GarageYrBlt" in df.columns and "YrSold" in df.columns:
            df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
            df["GarageAge"] = df["GarageAge"].clip(lower=0)
            self.new_features.append("GarageAge")
        return df

    def create_quality_features(self, df):
        df = df.copy()
        if "OverallQual" in df.columns and "OverallCond" in df.columns:
            df["QualCondProduct"] = df["OverallQual"] * df["OverallCond"]
            self.new_features.append("QualCondProduct")
        if "OverallQual" in df.columns and "GrLivArea" in df.columns:
            df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
            self.new_features.append("QualArea")
        return df

    def create_bathroom_features(self, df):
        df = df.copy()
        bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        existing = [c for c in bath_cols if c in df.columns]
        if existing:
            df["TotalBathrooms"] = sum(df.get(c, 0) for c in existing)
            self.new_features.append("TotalBathrooms")
        return df

    def log_transform_skewed(self, df, threshold=0.75):
        df = df.copy()
        numeric = df.select_dtypes(include=[np.number])
        skewed = numeric.apply(lambda x: x.skew()).abs()
        skewed_features = skewed[skewed > threshold].index
        for feat in skewed_features:
            if (df[feat] >= 0).all():
                df[f"{feat}_log"] = np.log1p(df[feat])
        return df

    def engineer(self, df):
        df = self.create_area_features(df)
        df = self.create_age_features(df)
        df = self.create_quality_features(df)
        df = self.create_bathroom_features(df)
        print(f"Created {len(self.new_features)} new features")
        return df
