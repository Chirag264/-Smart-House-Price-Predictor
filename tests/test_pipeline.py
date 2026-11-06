"""
Automated Unit and Integration Tests for House Price Prediction Pipeline
Run with: pytest -v
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import pytest

# Ensure root directory is on Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', 'Housing.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'house_price_model.pkl')

from src.pipeline import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_INPUT_FEATURES


@pytest.fixture(scope="module")
def dataset():
    """Fixture to load dataset."""
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df


@pytest.fixture(scope="module")
def model_pipeline():
    """Fixture to load trained model pipeline."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def test_dataset_structure(dataset):
    """Verify dataset columns and data integrity."""
    expected_columns = {
        'price', 'area', 'bedrooms', 'bathrooms', 'stories',
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'parking', 'preferarea', 'furnishingstatus'
    }
    assert set(dataset.columns) == expected_columns, "Dataset columns do not match expected structure"
    assert len(dataset) == 545, f"Expected 545 rows, found {len(dataset)}"
    assert dataset.isnull().sum().sum() == 0, "Dataset contains unexpected missing values"


def test_model_loaded(model_pipeline):
    """Verify that the loaded object is a Scikit-Learn Pipeline."""
    assert hasattr(model_pipeline, 'predict'), "Loaded pipeline does not implement predict()"
    assert 'preprocessor' in model_pipeline.named_steps, "Pipeline missing 'preprocessor' step"
    assert 'model' in model_pipeline.named_steps, "Pipeline missing 'model' step"


def test_single_prediction_format(model_pipeline):
    """Verify that a standard house input produces a positive realistic price."""
    sample = pd.DataFrame([{
        'area': 4500,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 'Yes',
        'guestroom': 'No',
        'basement': 'Yes',
        'hotwaterheating': 'No',
        'airconditioning': 'Yes',
        'parking': 1,
        'preferarea': 'Yes',
        'furnishingstatus': 'Semi-furnished'
    }])

    pred = model_pipeline.predict(sample)
    assert len(pred) == 1, "Expected single prediction"
    assert isinstance(pred[0], (float, np.floating, int, np.integer)), "Prediction is not numeric"
    assert 1_500_000 < pred[0] < 20_000_000, f"Suspicious prediction: {pred[0]}"


def test_batch_prediction(model_pipeline, dataset):
    """Verify that batch predictions run smoothly on dataset features."""
    X = dataset.drop('price', axis=1)
    preds = model_pipeline.predict(X)
    assert len(preds) == len(dataset), "Batch prediction output length mismatch"
    assert not np.isnan(preds).any(), "Batch predictions contain NaN values"


def test_furnishing_status_impact(model_pipeline):
    """Verify that furnished houses have higher or equal valuation than unfurnished houses."""
    base_specs = {
        'area': 5000,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 'Yes',
        'guestroom': 'No',
        'basement': 'No',
        'hotwaterheating': 'No',
        'airconditioning': 'Yes',
        'parking': 1,
        'preferarea': 'Yes',
    }

    df_furnished = pd.DataFrame([{**base_specs, 'furnishingstatus': 'Furnished'}])
    df_unfurnished = pd.DataFrame([{**base_specs, 'furnishingstatus': 'Unfurnished'}])

    price_furnished = model_pipeline.predict(df_furnished)[0]
    price_unfurnished = model_pipeline.predict(df_unfurnished)[0]

    assert price_furnished >= price_unfurnished, (
        f"Furnished (₹{price_furnished:,.0f}) lower than unfurnished (₹{price_unfurnished:,.0f})"
    )


def test_airconditioning_impact(model_pipeline):
    """Verify that AC presence adds value, holding all else equal."""
    base_specs = {
        'area': 5000,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 'Yes',
        'guestroom': 'No',
        'basement': 'No',
        'hotwaterheating': 'No',
        'parking': 1,
        'preferarea': 'Yes',
        'furnishingstatus': 'Furnished'
    }

    df_ac = pd.DataFrame([{**base_specs, 'airconditioning': 'Yes'}])
    df_no_ac = pd.DataFrame([{**base_specs, 'airconditioning': 'No'}])

    price_ac = model_pipeline.predict(df_ac)[0]
    price_no_ac = model_pipeline.predict(df_no_ac)[0]

    assert price_ac > price_no_ac, "Air conditioning did not add positive valuation"
