# 📋 Housing Dataset Dictionary

This dataset contains **545 verified residential sales records** with 13 features used to estimate fair market house prices.

---

## 📊 Summary of Features

| # | Feature Name | Data Type | Permitted Values / Range | Description | Real Estate Impact |
|---|---|---|---|---|---|
| 1 | **`price`** *(Target)* | Integer | ₹ 1,750,000 – ₹ 13,300,000 | Final residential transaction price in Indian Rupees ($\text{₹}$). | Model target variable to predict. |
| 2 | **`area`** | Integer | 1,650 – 16,200 sq. ft | Total carpet / built-up square footage of the property. | **Primary Price Driver** (~70% of valuation). |
| 3 | **`bedrooms`** | Integer | 1 – 6 | Number of bedrooms in the residence. | Structural layout driver. |
| 4 | **`bathrooms`** | Integer | 1 – 4 | Number of attached and common bathrooms. | High-value amenity; adds ~₹ 8–10 Lakhs each. |
| 5 | **`stories`** | Integer | 1 – 4 | Total number of building stories / vertical floors. | Measures multi-level architecture (e.g. Duplex, Triplex). |
| 6 | **`mainroad`** | Categorical | `Yes` / `No` | Whether the property directly faces or connects to a primary road. | Road access provides commercial and commuter convenience. |
| 7 | **`guestroom`** | Categorical | `Yes` / `No` | Dedicated guest bedroom available. | Comfort differentiator for large family homes. |
| 8 | **`basement`** | Categorical | `Yes` / `No` | Finished or usable underground basement. | Added utility space for storage, recreation, or parking. |
| 9 | **`hotwaterheating`** | Categorical | `Yes` / `No` | Dedicated boiler or geyser hot water circulation. | Climate utility amenity. |
| 10 | **`airconditioning`** | Categorical | `Yes` / `No` | Central or split air conditioning installed. | **Strong comfort driver** (adds ~₹ 8–12 Lakhs). |
| 11 | **`parking`** | Integer | 0 – 3 | Number of dedicated covered vehicle parking slots. | High value in dense urban settings. |
| 12 | **`preferarea`** | Categorical | `Yes` / `No` | Located in a prime, high-demand, or upscale neighborhood. | **Location premium**; adds significant value. |
| 13 | **`furnishingstatus`** | Categorical | `Furnished`, `Semi-furnished`, `Unfurnished` | Current furnishing state of the home interior. | Furnished commands ~15% premium over unfurnished. |

---

## 🔍 Data Source & Validation Notes
- **Total Records**: 545 rows, 13 columns.
- **Missing Values**: 0 missing values (100% complete).
- **Target Distribution**: Right-skewed distribution characteristic of free-market residential real estate.

