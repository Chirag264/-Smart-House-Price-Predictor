"""
Hyperparameter Tuning Module
Performs 5-Fold Cross-Validation Grid Search over Gradient Boosting hyperparameters.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import create_pipeline


def tune():
    data_path = os.path.join(BASE_DIR, 'data', 'Housing.csv')
    model_dir = os.path.join(BASE_DIR, 'model')
    model_path = os.path.join(model_dir, 'house_price_model.pkl')

    print("=" * 60)
    print("🎯 STARTING HYPERPARAMETER OPTIMIZATION (GRID SEARCH CV)")
    print("=" * 60)

    df = pd.read_csv(data_path)
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_pipeline = create_pipeline()

    param_grid = {
        'model__n_estimators': [100, 150, 200],
        'model__learning_rate': [0.03, 0.05, 0.08, 0.1],
        'model__max_depth': [3, 4],
        'model__subsample': [0.8, 1.0]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Searching parameter grid with 5-fold CV...")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n🏆 BEST HYPERPARAMETERS FOUND:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param.replace('model__', '')}: {value}")

    print(f"\nBest Cross-Validation R²: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    print(f"Test MAE:  ₹ {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"Test RMSE: ₹ {root_mean_squared_error(y_test, y_pred):,.2f}")
    print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")

    print("\nRetraining on full dataset and saving...")
    best_pipeline.fit(X, y)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    print(f"✓ Saved to: {model_path}")


if __name__ == '__main__':
    tune()

