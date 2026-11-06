<div align="center">

# 🏠 Smart House Price Predictor
### End-to-End Machine Learning System & Interactive Valuation Web App

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Build-Production--Ready-success)]()

<p align="center">
  <b>Estimate fair market residential property valuations with machine learning.</b><br>
  Built with clean Scikit-Learn pipelines, 545 verified sales records, automated pytest validation, and a modern, beginner-friendly Streamlit web app with mortgage calculations and upgrade simulators.
</p>

[Explore Web App](#-quickstart-guide) •
[ML Workflow](#-machine-learning-architecture) •
[Model Benchmarks](#-model-evaluation--benchmarks) •
[Project Structure](#-clean-project-structure) •
[Data Dictionary](#-dataset-dictionary)

---

</div>

## 🌟 Key Highlights & Features

- 🧠 **Trained on 545 Verified Sales**: Full market diversity spanning ₹ 17.5 Lakhs starter flats up to ₹ 1.33 Crore luxury estates.
- ⚡ **1-Click Quick Templates**: Instant auto-fill buttons for **2 BHK Budget Flat**, **3 BHK Family Home**, and **4 BHK Luxury Villa**—built for non-technical users!
- 🏙️ **Regional Metro Indexing**: Real-time city multipliers for Mumbai, Bengaluru, Delhi NCR, Hyderabad, Pune, Chennai, Ahmedabad, and Tier-2 hubs.
- 🧭 **Interactive Plotly Speedometer**: Visual affordability gauge and market percentile distribution chart.
- 💳 **Built-in Mortgage & Monthly EMI Calculator**: Instantly calculates down payments, loan tenures, and monthly bank EMIs.
- 💡 **Value Upgrade Simulator (What-If Analysis)**: Live financial impact analysis of adding Air Conditioning (+₹ 8.5L), Furnishing (+₹ 4.2L), or moving to a Prime Area.
- 📥 **1-Click Downloadable Valuation Report**: Generates a clean, timestamped text receipt of property specs, price range, and financing estimates.
- 🔒 **Leak-Free Scikit-Learn Pipeline**: Unified `ColumnTransformer` handles missing values (median/mode), feature scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`).
- 🧪 **Automated Testing Suite**: 6 automated `pytest` tests validating dataset constraints, model serialization, prediction boundaries, and feature monotonicity.

---

## 🖥️ Application Interface Preview

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ✨ AI-Powered Real Estate Valuation                                             │
│ 🏠 Smart House Price Predictor                                                  │
│ Planning to buy, sell, or evaluate a home? Get an instant market valuation!     │
└─────────────────────────────────────────────────────────────────────────────────┘

  ⚡ Quick Templates:
  [ 🏠 2 BHK Budget Flat ]   [ 🏡 3 BHK Family Home ]   [ 🏰 4 BHK Luxury Villa ]

┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ 📐 Size & Rooms        │ │ ❄️ Comfort             │ │ 📍 Location & Region   │
│ • Area: 4,500 sq.ft    │ │ • Parking: 1 Car       │ │ • City: Mumbai (1.55x) │
│ • Bedrooms: 3 Beds     │ │ • AC: Yes              │ │ • Furnishing: Semi     │
│ • Bathrooms: 2 Baths   │ │ • Heating: No          │ │ • Prime Area: Yes      │
│ • Floors: 2 Stories    │ │ • Guest Room: No       │ │ • Main Road: Yes       │
└────────────────────────┘ └────────────────────────┘ └────────────────────────┘

                     [ 🔮 Calculate Estimated House Price ]

┌─────────────────────────────────────────────────────────────────────────────────┐
│                  Estimated Fair Market Valuation (Mumbai)                       │
│                               ₹ 87,42,000                                       │
│          ₹ 87.42 Lakhs • (₹ 0.87 Crore) • 🏰 Premium Luxury Segment             │
│          Realistic Range: ₹ 80.4 Lakhs — ₹ 94.4 Lakhs (±8% Market Variance)     │
└─────────────────────────────────────────────────────────────────────────────────┘

 [ Rate: ₹ 1,943 / sq.ft ]  [ AC: Included ]  [ Furnishing: Semi ]  [ City: Mumbai ]

  ▶ 🧭 Where You Rank Among 545 Sales (Top 18% of Properties)
  ▶ 💳 Home Loan & Monthly EMI Calculator (e.g. ₹ 60,650 / month @ 8.5% for 20 Yrs)
  ▶ 💡 Value Upgrade Simulator (AC, Furnishing, Prime Area)
  ▶ 📊 AI Model Transparency: Top 10 Factors Influencing Price
  ▶ 📥 Download Valuation Summary (TXT)
  ▶ ❓ Frequently Asked Questions (FAQ for Home Buyers)
```

---

## 📁 Clean Project Structure

Designed according to professional data science modular architecture:

```text
house-price-prediction/
│
├── app.py                           # 🌐 Main Streamlit interactive web application
├── House_Price_Prediction.ipynb     # 📓 Complete 9-section Jupyter notebook with outputs
├── train_model.py                   # ⚡ Quick model training entry point
├── run_app.bat                      # 🚀 1-Click Windows desktop application launcher
├── requirements.txt                 # 📦 Python project dependencies
├── README.md                        # 📖 Project documentation and guide
│
├── data/                            # 📊 Data directory
│   ├── Housing.csv                  # Clean housing dataset (545 sales records, 13 features)
│   └── data_dictionary.md          # 📋 Complete data dictionary & real estate impact guide
│
├── model/                           # 🧠 Serialized ML model artifacts
│   └── house_price_model.pkl        # Serialized pipeline (ColumnTransformer + GradientBoosting)
│
├── src/                             # ⚙️ Core modular Python source package
│   ├── __init__.py                  # Package initializer
│   ├── pipeline.py                  # Single source of truth for features & preprocessors
│   ├── train.py                     # Full model training, evaluation & serialization script
│   └── tune.py                      # 5-fold cross-validation GridSearch hyperparameter optimizer
│
└── tests/                           # 🧪 Automated quality assurance test suite
    ├── __init__.py
    └── test_pipeline.py             # 6 automated test cases (pytest)
```

---

## ⚙️ Machine Learning Architecture

```
┌─────────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
│  data/Housing.csv   │ ───> │ Exploratory Analysis    │ ───> │ ColumnTransformer      │
│  (545 Sales Records)│      │ Distribution & Heatmaps │      │ Imputer + Scaler + OHE │
└─────────────────────┘      └─────────────────────────┘      └───────────┬────────────┘
                                                                          │
┌─────────────────────┐      ┌─────────────────────────┐      ┌───────────▼────────────┐
│ Streamlit Web App   │ <─── │ model/                  │ <─── │ Gradient Boosting      │
│ with What-If Sim    │      │ house_price_model.pkl   │      │ Model Training & Eval  │
│ (app.py)            │      │ (Joblib Pipeline)       │      │ (R² = 0.67, MAE ₹9.6L) │
└─────────────────────┘      └─────────────────────────┘      └────────────────────────┘
```

### Preprocessing Pipeline (`src/pipeline.py`)
- **Numerical Features** (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`):
  - Missing Value Imputation: `SimpleImputer(strategy='median')`
  - Feature Scaling: `StandardScaler()` (mean = 0, variance = 1)
- **Categorical Features** (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `preferarea`, `furnishingstatus`):
  - Missing Value Imputation: `SimpleImputer(strategy='most_frequent')`
  - Categorical Encoding: `OneHotEncoder(handle_unknown='ignore')`

---

## 📊 Model Evaluation & Benchmarks

All models were evaluated on an **80/20 train-test split** (436 training houses, 109 holdout test houses) using 3 standard regression metrics:
1. **Mean Absolute Error (MAE)**: Average magnitude of prediction error in Indian Rupees ($\text{₹}$). Lower is better.
2. **Root Mean Squared Error (RMSE)**: Penalizes large outlier errors. Lower is better.
3. **$R^2$ Score (Coefficient of Determination)**: Proportion of variance in house prices explained by the features ($1.0$ is perfect). Higher is better.

| Model | Test MAE ($\text{₹}$) | Test RMSE ($\text{₹}$) | Test $R^2$ Score | Characteristics |
|---|---|---|---|---|
| **Gradient Boosting Regressor** | **₹ 9,67,227** | **₹ 12,99,730** | **0.6658** | 🏆 **Best Performer**: Sequential boosting trees with lowest error. |
| **Linear Regression** | ₹ 9,70,043 | ₹ 13,24,507 | 0.6529 | Baseline Parametric Model: Strictly monotonic across all features. |
| **Random Forest Regressor** | ₹ 10,25,961 | ₹ 14,03,925 | 0.6101 | Bagging Ensemble: Parallel average of 100 decision trees. |

---

## 🎯 Top Factors Influencing Property Valuation

Extracted directly from the trained Gradient Boosting model's `feature_importances_`:

| Rank | Feature | Importance Weight | Real Estate Rationale |
|---|---|---|---|
| 1 | **Area (sq. ft)** | **44.6%** | Total carpet area is the primary determinant of land & construction costs. |
| 2 | **Bathrooms** | **17.5%** | Plumbing infrastructure & bathroom count heavily dictate luxury tiers. |
| 3 | **Parking Spaces** | **5.9%** | Dedicated covered parking slots are at a premium in urban markets. |
| 4 | **Air Conditioning** | **5.8%** | Installed split/central AC commands an immediate comfort premium. |
| 5 | **Stories (Floors)** | **5.5%** | Multi-level homes (Duplex, Triplex) reflect higher square footage & privacy. |
| 6 | **Bedrooms** | **3.7%** | Core structural requirement for family residency. |
| 7 | **Furnishing Status** | **3.1%** | Ready-to-move furnished interiors save buyers significant setup time. |
| 8 | **Basement** | **1.9%** | Additional utility space for storage, recreation, or home offices. |
| 9 | **Preferred Area** | **1.8%** | Prime, high-demand neighborhood location premium. |
| 10 | **Main Road Access** | **1.2%** | Direct road connectivity for transport convenience. |

---

## 📋 Dataset Dictionary

Detailed reference for the 13 features in `data/Housing.csv`:

| Feature | Type | Range / Permitted Values | Description |
|---|---|---|---|
| `price` *(Target)* | Integer | ₹ 17,50,000 – ₹ 1,33,00,000 | Final residential transaction value in Rupees ($\text{₹}$). |
| `area` | Integer | 1,650 – 16,200 sq. ft | Total built-up / carpet area of the residence. |
| `bedrooms` | Integer | 1 – 6 | Number of bedrooms. |
| `bathrooms` | Integer | 1 – 4 | Number of attached and common bathrooms. |
| `stories` | Integer | 1 – 4 | Number of building levels/stories. |
| `parking` | Integer | 0 – 3 | Dedicated vehicle parking spaces. |
| `mainroad` | Categorical | `Yes` / `No` | Direct connection or frontage to a primary thoroughfare. |
| `guestroom` | Categorical | `Yes` / `No` | Dedicated guest bedroom. |
| `basement` | Categorical | `Yes` / `No` | Finished usable underground basement space. |
| `hotwaterheating`| Categorical | `Yes` / `No` | Boiler / geyser hot water circulation system. |
| `airconditioning`| Categorical | `Yes` / `No` | Central or split air conditioning installed. |
| `preferarea` | Categorical | `Yes` / `No` | Located in a prime, high-demand neighborhood. |
| `furnishingstatus`| Categorical| `Furnished`, `Semi-furnished`, `Unfurnished`| Interior furnishing level. |

*(For full details, see [`data/data_dictionary.md`](data/data_dictionary.md))*

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.8+ (Python 3.10, 3.11, 3.12, or 3.14 tested)
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Chirag264/-Smart-House-Price-Predictor.git
cd -Smart-House-Price-Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
> **Windows Users**: You can also double-click **`run_app.bat`** to launch the web app instantly without touching the command prompt!

Access the interactive dashboard in your browser at:
```text
http://localhost:8501
```

---

## 🧪 Running Automated Tests

The project includes unit and integration tests under `tests/test_pipeline.py`. Run them with `pytest`:

```bash
pytest -v
```

Expected output:
```text
============================= test session starts =============================
tests/test_pipeline.py::test_dataset_structure PASSED                    [ 16%]
tests/test_pipeline.py::test_model_loaded PASSED                         [ 33%]
tests/test_pipeline.py::test_single_prediction_format PASSED             [ 50%]
tests/test_pipeline.py::test_batch_prediction PASSED                     [ 66%]
tests/test_pipeline.py::test_furnishing_status_impact PASSED             [ 83%]
tests/test_pipeline.py::test_airconditioning_impact PASSED               [100%]
============================== 6 passed in 2.30s ==============================
```

---

## ⚡ Developer Commands

| Task | Command |
|---|---|
| **Run Streamlit Web App** | `streamlit run app.py` (or click `run_app.bat`) |
| **Retrain ML Model** | `python train_model.py` (or `python src/train.py`) |
| **Run Hyperparameter Tuning** | `python src/tune.py` |
| **Run Test Suite** | `pytest -v` |
| **Open Jupyter Notebook** | `jupyter lab House_Price_Prediction.ipynb` |

---

## 🌐 Deploying to Streamlit Community Cloud (Free)

You can share your live web application online for free:

1. Push this repository to your **GitHub** account:
   ```bash
   git init
   git add .
   git commit -m "feat: complete end-to-end house price prediction system"
   git branch -M main
   git remote add origin https://github.com/Chirag264/-Smart-House-Price-Predictor.git
   git push -u origin main
   ```
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to:
   ```text
   app.py
   ```
4. Click **Deploy!** Your app will be live on the web in under 2 minutes with a public URL to share on your resume, LinkedIn, or portfolio.

---

## 💡 Troubleshooting & FAQs

<details>
<summary><b>Q1: Why does square footage have the largest weight in valuation?</b></summary>
<br>
In residential real estate, land acquisition costs and civil construction rates scale directly with built-up square footage. An extra 500 sq. ft adds far more structural value than cosmetic fittings.
</details>

<details>
<summary><b>Q2: "Model file not found" error when launching Streamlit?</b></summary>
<br>
Run <code>python train_model.py</code> in your terminal. This trains the Gradient Boosting pipeline on <code>data/Housing.csv</code> and serializes it to <code>model/house_price_model.pkl</code>.
</details>

<details>
<summary><b>Q3: Windows execution alias error (`python.exe: The system cannot find the path specified`)?</b></summary>
<br>
Windows Store aliases can conflict with standard Python installations. Disable the "App execution alias for Python" in Windows Settings, or run using <code>run_app.bat</code> which automatically locates your local Python environment.
</details>

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE). You are free to use, modify, and build upon this work for personal, educational, or commercial projects.

---

<div align="center">

Made with ❤️ using **Python**, **Scikit-learn**, **Streamlit**, and **Plotly**

⭐ If you found this project helpful, please consider starring this repository!

</div>
"# -Smart-House-Price-Predictor" 
