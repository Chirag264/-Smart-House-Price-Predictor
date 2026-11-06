"""
Model Training Module
Trains the machine learning pipeline on data/Housing.csv and saves to model/house_price_model.pkl.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Import centralized pipeline builder
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import create_pipeline


def train():
    data_path = os.path.join(BASE_DIR, 'data', 'Housing.csv')
    model_dir = os.path.join(BASE_DIR, 'model')
    model_path = os.path.join(model_dir, 'house_price_model.pkl')

    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"📊 Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop('price', axis=1)
    y = df['price']

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model pipeline
    pipeline = create_pipeline()

    print("⚙️ Training Gradient Boosting Pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Test Set Evaluation:")
    print(f"  • MAE:      ₹ {mae:,.2f}")
    print(f"  • RMSE:     ₹ {rmse:,.2f}")
    print(f"  • R² Score: {r2:.4f}")

    # Retrain on full dataset for maximum deployment accuracy
    print("\n🔄 Retraining best pipeline on full 545 samples for deployment...")
    pipeline.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"✓ Model successfully saved to: {model_path}")


if __name__ == '__main__':
    train()

