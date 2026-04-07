from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


# 1. Cleaner for null flavors and string mappings (Steps 5 & 6)
class CarDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.null_flavors = {"", "null", "NaN", "NULL", "none", "None"}
        self.mapping = {"5more": "6", "more": "6"}

    def fit(self, X: Any, y: Any = None) -> "CarDataCleaner":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        # 1. Ensure we work on a DataFrame (not numpy array)
        X = pd.DataFrame(X).copy()

        # 2. Globally replace chaos nulls
        X = X.replace(self.null_flavors, np.nan)

        # 3. Specific strings replacements for doors/seats
        if "doors" in X.columns:
            X["doors"] = X["doors"].replace("5more", "6")

        if "seats" in X.columns:
            X["seats"] = X["seats"].replace("more", "6")

        return X

    # 2. Imputer for missing values


class CarDataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.int_cols = ["doors", "seats"]
        self.cat_cols = ["price", "maintenance", "safety", "storage"]
        self.output_features_: list[str] = []
        # Standard ColumnTransformer logic
        self.ct = ColumnTransformer(
            transformers=[
                ("median_imputer", SimpleImputer(strategy="median"), self.int_cols),
                (
                    "mode_imputer",
                    SimpleImputer(strategy="most_frequent"),
                    self.cat_cols,
                ),
            ],
            remainder="drop",
        )

    def fit(self, X: Any, y: Any = None) -> "CarDataImputer":
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()
        self.ct.fit(X_df)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        X_df = pd.DataFrame(X, columns=self.feature_names)
        # Note: ColumnTransformer reorders columns (int_cols first, then cat_cols)
        # We must track this new order for the next step in the pipeline
        transformed_data = self.ct.transform(X_df)

        # FIX: Dynamically determine output columns
        # ColumnTransformer places transformed columns FIRST, then 'passthrough' columns
        # If there's a 7th column, we need to account for it or drop it
        if transformed_data.shape[1] == 7:
            # If you have an extra column (like a label or ID), we name it 'extra'
            # to avoid the shape mismatch error
            self.output_features_ = self.int_cols + self.cat_cols + ["unknown_feature"]
        else:
            self.output_features_ = self.int_cols + self.cat_cols

        return pd.DataFrame(transformed_data, columns=self.output_features_)


# 3. Ordinal Mapping for Car Features
class CarDataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        # Define the columns and their specific logical order
        self.features_map: dict[str, list[str]] = {
            "price": ["low", "med", "high", "vhigh"],
            "maintenance": ["low", "med", "high", "vhigh"],
            "storage": ["small", "med", "big"],
            "safety": ["low", "med", "high"],
        }

        # Initialize the encoder with your specific categories
        self.cat_cols = list(self.features_map.keys())
        self.category_lists = list(self.features_map.values())
        self.encoder = OrdinalEncoder(
            categories=self.category_lists, handle_unknown="use_encoded_value", unknown_value=-1
        )

    def fit(self, X: Any, y: Any = None) -> "CarDataEncoder":
        # Ensure X is a DataFrame to select columns by name
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()
        # Fit the encoder only on the categorical columns
        self.encoder.fit(X_df[self.cat_cols])
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        # Create a copy to avoid modifying the original data
        X_df = pd.DataFrame(X).copy()

        # 1. Transform the categorical columns using the fitted encoder
        X_df[self.cat_cols] = self.encoder.transform(X_df[self.cat_cols])

        # 2. Final numeric check for doors/seats (Step 7 & 11)
        # Since '5more' was changed to '6' in the cleaner, we convert to numeric here
        for col in ["doors", "seats"]:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        return X_df
