"""Unit tests for regression models."""

import unittest
import numpy as np
from sklearn.datasets import make_regression
from src.models import HousePriceModels


class TestHousePriceModels(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        self.models = HousePriceModels()

    def test_linear_regression(self):
        model = self.models.train_linear_regression(self.X, self.y)
        preds = self.models.predict("linear_regression", self.X)
        self.assertEqual(len(preds), 100)

    def test_random_forest(self):
        model = self.models.train_random_forest(self.X, self.y, n_estimators=10)
        preds = self.models.predict("random_forest", self.X)
        self.assertEqual(len(preds), 100)

    def test_feature_importance(self):
        self.models.train_random_forest(self.X, self.y, n_estimators=10)
        imp = self.models.get_feature_importance("random_forest")
        self.assertEqual(len(imp), 10)


if __name__ == "__main__":
    unittest.main()
