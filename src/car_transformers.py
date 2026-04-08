from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class CarDataCleaner(BaseEstimator, TransformerMixin):
    """
    Standardizes null values and string mappings in car dataset.

    Replaces various null flavor representations ("", "null", "NaN", "NULL", "none", "None")
    with np.nan for consistent handling. Also maps string values like "5more" to numeric
    equivalents ("6") for categorical features.

    Parameters
    ----------
    feature_names : list[str]
        List of feature column names in the dataset.

    Attributes
    ----------
    feature_names : list[str]
        Stored feature names for reference.
    null_flavors : set
        Set of null value representations to be replaced with np.nan.
    mapping : dict[str, str]
        Dictionary mapping specific string values to their replacements.

    Examples
    --------
    >>> cleaner = CarDataCleaner(feature_names=['price', 'doors', 'seats'])
    >>> X = pd.DataFrame({'price': ['low', ''], 'doors': ['5more', '3']})
    >>> X_clean = cleaner.fit_transform(X)
    """

    def __init__(self, feature_names: list[str]) -> None:
        """
        Initialize the CarDataCleaner.

        Parameters
        ----------
        feature_names : list[str]
            List of feature column names in the dataset.
        """
        self.feature_names = feature_names
        # Use a list for null flavor tokens so callers can safely pass this
        # sequence into pandas.replace(...) without needing to cast.
        self.null_flavors = ["", "null", "NaN", "NULL", "none", "None"]
        self.mapping = {"5more": "6", "more": "6"}

    def fit(self, X: pd.DataFrame | dict | list, y: Any = None) -> "CarDataCleaner":
        """
        Fit the transformer (no-op for this transformer).

        Parameters
        ----------
        X : pd.DataFrame, dict, or array-like
            Input data (not used, present for API consistency).
        y : None, optional
            Ignored. Present here for API consistency by convention.

        Returns
        -------
        CarDataCleaner
            Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame | dict | list) -> pd.DataFrame:
        """
        Clean null flavors and apply string mappings.

        Replaces null flavor representations with np.nan and applies specific string
        mappings (e.g., "5more" -> "6" for doors, "more" -> "6" for seats).

        Parameters
        ----------
        X : pd.DataFrame, dict, or array-like
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with standardized null values and string mappings applied.
        """
        # 1. Ensure we work on a DataFrame (not numpy array)
        X_df = pd.DataFrame(X).copy()

        # 2. Standardize null-like string tokens without using replace downcasting paths
        X_df = X_df.mask(X_df.isin(self.null_flavors), np.nan)

        # 3. Specific strings replacements for doors/seats (warning-safe mapping)
        if "doors" in X_df.columns:
            X_df["doors"] = X_df["doors"].map(lambda v: self.mapping.get(v, v))

        if "seats" in X_df.columns:
            X_df["seats"] = X_df["seats"].map(lambda v: self.mapping.get(v, v))

        return X_df


class CarDataImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using different strategies for numeric and categorical features.

    Uses median imputation for numeric columns (doors, seats) and mode (most frequent value)
    imputation for categorical columns (price, maintenance, safety, storage).

    Parameters
    ----------
    feature_names : list[str]
        List of feature column names in the dataset.

    Attributes
    ----------
    feature_names : list[str]
        Stored feature names for reference.
    int_cols : list[str]
        Numeric columns to be imputed with median strategy: ["doors", "seats"].
    cat_cols : list[str]
        Categorical columns to be imputed with mode strategy:
        ["price", "maintenance", "safety", "storage"].
    ct : ColumnTransformer
        Scikit-learn ColumnTransformer for applying different strategies to different columns.
    output_features_ : list[str]
        List of output feature names after transformation (set during transform).

    Examples
    --------
    >>> imputer = CarDataImputer(feature_names=['price', 'doors', 'seats', 'safety'])
    >>> X = pd.DataFrame({'price': ['low', None], 'doors': [2, None], 'seats': [4, 2], 'safety': ['high', None]})
    >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(self, feature_names: list[str]) -> None:
        """
        Initialize the CarDataImputer.

        Parameters
        ----------
        feature_names : list[str]
            List of feature column names in the dataset.
        """
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

    def fit(self, X: pd.DataFrame | dict | list, y: Any = None) -> "CarDataImputer":
        """
        Fit the imputer on the data.

        Learns median and mode values from the input data for imputation.

        Parameters
        ----------
        X : pd.DataFrame, dict, or array-like
            Input data to fit the imputer on.
        y : None, optional
            Ignored. Present here for API consistency by convention.

        Returns
        -------
        CarDataImputer
            Returns self.
        """
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()
        self.ct.fit(X_df)
        return self

    def transform(self, X: pd.DataFrame | dict | list) -> pd.DataFrame:
        """
        Impute missing values in the input data.

        Replaces missing values with learned median (numeric columns) or
        mode (categorical columns) from the fitted data.

        Parameters
        ----------
        X : pd.DataFrame, dict, or array-like
            Input data to impute.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with imputed values. Columns are ordered as:
            [numeric columns, categorical columns].

        Notes
        -----
        The ColumnTransformer reorders columns: numeric columns first, then categorical.
        """
        X_df = pd.DataFrame(X, columns=self.feature_names)
        transformed_data = self.ct.transform(X_df)

        raw_feature_names = list(self.ct.get_feature_names_out())
        # get_feature_names_out returns names like "median_imputer__doors".
        self.output_features_ = [name.split("__", 1)[-1] for name in raw_feature_names]

        transformed_data_array = np.asarray(transformed_data)

        return pd.DataFrame(transformed_data_array, columns=self.output_features_)


class CarDataEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features using ordinal encoding while preserving feature hierarchy.

    Converts categorical columns to ordinal (integer) values based on predefined orderings.
    For example, price is encoded as: low=0, med=1, high=2, vhigh=3. This preserves the
    natural ordering of these features, which is important for models like KNN and SVM.

    Parameters
    ----------
    feature_names : list[str]
        List of feature column names in the dataset.

    Attributes
    ----------
    feature_names : list[str]
        Stored feature names for reference.
    features_map : dict[str, list[str]]
        Dictionary mapping feature names to their ordered category values.
    cat_cols : list[str]
        List of categorical columns to be encoded.
    category_lists : list[list[str]]
        List of ordered category lists for each categorical feature.
    encoder : OrdinalEncoder
        Scikit-learn OrdinalEncoder for performing the ordinal encoding.

    Examples
    --------
    >>> encoder = CarDataEncoder(feature_names=['price', 'safety', 'doors'])
    >>> X = pd.DataFrame({'price': ['low', 'high'], 'safety': ['low', 'high'], 'doors': [2, 4]})
    >>> X_encoded = encoder.fit_transform(X)
    # price: low->0, high->2; safety: low->0, high->2; doors unchanged
    """

    def __init__(self, feature_names: list[str]) -> None:
        """
        Initialize the CarDataEncoder.

        Parameters
        ----------
        feature_names : list[str]
            List of feature column names in the dataset.
        """
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

    def fit(self, X: pd.DataFrame | dict | list, y: Any = None) -> "CarDataEncoder":
        """
        Fit the encoder on the data.

        Validates that all categorical columns are present and have expected categories.

        Parameters
        ----------
        X : pd.DataFrame, dict, or array-like
            Input data to fit the encoder on.
        y : None, optional
            Ignored. Present here for API consistency by convention.

        Returns
        -------
        CarDataEncoder
            Returns self.
        """
        # Ensure X is a DataFrame to select columns by name
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()
        # Fit the encoder only on the categorical columns
        self.encoder.fit(X_df[self.cat_cols])
        return self

    def transform(self, X: pd.DataFrame | dict | list) -> pd.DataFrame:
        """
        Encode categorical features to ordinal values.

        Converts categorical columns to integers based on the predefined ordering.
        Also converts "doors" and "seats" columns to numeric type.

        Parameters
        ----------
        X : pd.DataFrame, dict, or array-like
            Input data to encode.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with categorical features encoded to ordinal values.
            Unknown categories are encoded as -1.

        Notes
        -----
        The predefined orderings preserve natural hierarchies:
        - price: low(0) < med(1) < high(2) < vhigh(3)
        - maintenance: low(0) < med(1) < high(2) < vhigh(3)
        - storage: small(0) < med(1) < big(2)
        - safety: low(0) < med(1) < high(2)
        """
        # Create a copy to avoid modifying the original data
        X_df = pd.DataFrame(X).copy()

        # 1. Transform the categorical columns using the fitted encoder
        X_df[self.cat_cols] = self.encoder.transform(X_df[self.cat_cols])

        # 2. Final numeric check for doors/seats (Step 7 & 11)
        # Since '5more' was changed to '6' in the cleaner, we convert to numeric here
        for col in ["doors", "seats"]:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        return X_df
