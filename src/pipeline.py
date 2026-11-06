"""
Smart House Price Predictor - Core Pipeline Module
Centralizes feature columns, preprocessors, and Scikit-Learn pipelines
to prevent duplicate code and ensure 100% consistency between training and prediction.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature columns
NUMERICAL_FEATURES = [
    'area',
    'bedrooms',
    'bathrooms',
    'stories',
    'parking'
]

CATEGORICAL_FEATURES = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'preferarea',
    'furnishingstatus'
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer combining:
    1. Numerical: SimpleImputer(median) + StandardScaler
    2. Categorical: SimpleImputer(most_frequent) + OneHotEncoder(ignore unseen)
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def create_pipeline(regressor=None) -> Pipeline:
    """
    Creates an end-to-end Pipeline connecting preprocessing with a regressor.
    Defaults to GradientBoostingRegressor(random_state=42).
    """
    if regressor is None:
        regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('model', regressor)
    ])

    return pipeline

