"""Unit tests for preprocessing module."""

import unittest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "num1": [1.0, np.nan, 3.0, 4.0],
            "num2": [10, 20, 30, 40],
            "cat1": ["a", "b", None, "a"],
        })
        self.preprocessor = DataPreprocessor()

    def test_fit_transform(self):
        self.preprocessor.fit(self.df)
        result = self.preprocessor.transform(self.df)
        self.assertFalse(result.isnull().any().any())

    def test_outlier_handling(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        self.preprocessor.numeric_cols = ["val"]
        result = self.preprocessor.handle_outliers(df, columns=["val"])
        self.assertLess(result["val"].max(), 100)

    def test_preserves_shape(self):
        self.preprocessor.fit(self.df)
        result = self.preprocessor.transform(self.df)
        self.assertEqual(result.shape, self.df.shape)


if __name__ == "__main__":
    unittest.main()
