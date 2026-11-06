"""
Smart House Price Predictor - Setup Verification Script
Quickly validates your local environment, dependencies, dataset, and trained model.
Run with: python verify_setup.py
"""

import os
import sys

def verify():
    print("=" * 60)
    print("🔍 VERIFYING SMART HOUSE PRICE PREDICTOR SETUP")
    print("=" * 60)

    # 1. Check Python version
    py_version = sys.version_info
    print(f"🐍 Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required.")
        return False
    print("   ✓ Python version compatible.")

    # 2. Check Dependencies
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('joblib', 'joblib'),
        ('pytest', 'pytest')
    ]

    print("\n📦 Checking Required Dependencies:")
    missing = []
    for pkg_name, import_name in required_packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, '__version__', 'installed')
            print(f"   ✓ {pkg_name} ({ver})")
        except ImportError:
            print(f"   ❌ Missing: {pkg_name}")
            missing.append(pkg_name)

    if missing:
        print(f"\n❌ Please install missing dependencies using: pip install {' '.join(missing)}")
        return False

    # 3. Check Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'Housing.csv')
    print(f"\n📊 Checking Dataset: {data_path}")
    if not os.path.exists(data_path):
        print("❌ Error: Housing.csv not found in data/ directory.")
        return False

    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"   ✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    if len(df) != 545:
        print(f"   ⚠️ Warning: Expected 545 rows, found {len(df)}.")

    # 4. Check Model
    model_path = os.path.join(base_dir, 'model', 'house_price_model.pkl')
    print(f"\n🧠 Checking Serialized Model: {model_path}")
    if not os.path.exists(model_path):
        print("❌ Warning: house_price_model.pkl not found. Training it now...")
        from src.train import train
        train()

    import joblib
    pipeline = joblib.load(model_path)
    print("   ✓ Model loaded successfully.")

    # 5. Test Inference
    print("\n🧪 Testing Sample Inference:")
    sample = pd.DataFrame([{
        'area': 4500, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
        'mainroad': 'Yes', 'guestroom': 'No', 'basement': 'Yes',
        'hotwaterheating': 'No', 'airconditioning': 'Yes',
        'parking': 1, 'preferarea': 'Yes', 'furnishingstatus': 'Semi-furnished'
    }])
    pred = pipeline.predict(sample)[0]
    print(f"   ✓ Test Prediction: ₹ {pred:,.2f} (~₹ {pred/100000:.2f} Lakhs)")

    print("\n" + "=" * 60)
    print("🎉 ALL SYSTEMS OPERATIONAL! Your project is 100% ready.")
    print("To launch the web app, run: streamlit run app.py")
    print("Or on Windows, double-click: run_app.bat")
    print("=" * 60)
    return True

if __name__ == '__main__':
    verify()
