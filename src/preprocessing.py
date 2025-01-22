"""Data preprocessing pipeline for house price prediction."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handles missing values, encoding, and scaling."""

    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, X):
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if self.numeric_cols:
            self.numeric_imputer.fit(X[self.numeric_cols])
        if self.categorical_cols:
            self.categorical_imputer.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_cols:
            X[self.numeric_cols] = self.numeric_imputer.transform(X[self.numeric_cols])
        if self.categorical_cols:
            X[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])
            for col in self.categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    known = set(self.label_encoders[col].classes_)
                    X[col] = X[col].map(lambda v: v if v in known else "unknown")
                    if "unknown" not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, "unknown")
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X

    def handle_outliers(self, X, columns=None, method="iqr", factor=1.5):
        X = X.copy()
        cols = columns or self.numeric_cols
        for col in cols:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                X[col] = X[col].clip(lower, upper)
        return X

    def scale(self, X):
        X = X.copy()
        num_cols = [c for c in self.numeric_cols if c in X.columns]
        X[num_cols] = self.scaler.fit_transform(X[num_cols])
        return X
