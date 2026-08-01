# 🌾 Farm Yield Prediction

An end-to-end regression project that predicts crop yield (tons per hectare)
from farming and environmental data. The project covers the full pipeline —
data cleaning, exploratory data analysis, preprocessing, model building, and
regularization — and includes an interactive Streamlit demo for exploring
predictions.

## 📊 Dataset

343 farm records with 9 features and 1 target:

| Feature | Description |
|---|---|
| `fertilizer_kg_per_hectare` | Fertilizer applied per hectare |
| `rainfall_mm` | Rainfall received (mm) |
| `soil_ph` | Soil pH level |
| `farm_size_hectares` | Size of the farm |
| `years_farming_experience` | Farmer's years of experience |
| `crop_type` | Wheat, Corn, Barley, Rice |
| `irrigation_type` | Flood, Drip, Sprinkler |
| `region` | Upper Egypt, Sinai, Delta |
| `yield_tons_per_hectare` | **Target** — crop yield per hectare |

The raw data contained missing values, inconsistent category labels, and
outliers, making it a realistic cleaning exercise.

## 🧹 Data cleaning & preparation

- **Outlier detection** — visualized all numeric features with boxplots to
  spot extreme values.
- **Negative values** — `farm_size_hectares` contained invalid negative
  entries, corrected by taking the absolute value.
- **Inconsistent categories** — `crop_type` had inconsistent casing and
  whitespace (e.g. `"WHEAT"` vs `"Wheat"` vs `" wheat "`); standardized by
  stripping whitespace and capitalizing.
- **Duplicates** — removed exact duplicate rows.
- **Outlier capping** — applied IQR-based capping (values clipped to
  `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`) across all numeric columns.
- **Missing values** — `rainfall_mm` and `soil_ph` had missing entries,
  imputed with the column mean.

## 🔍 Exploratory data analysis

- Scatter plots of each feature against `yield_tons_per_hectare` showed
  `fertilizer_kg_per_hectare` has the most noticeable **non-linear**
  relationship with yield — a strong early signal that a linear model alone
  would underfit.
- A correlation heatmap was used to examine relationships between all
  numeric/encoded features and the target.

## ⚙️ Preprocessing

- **Label encoding** applied to the three categorical features
  (`crop_type`, `irrigation_type`, `region`), with each column's encoder
  saved for later reuse.
- **Train/test split** — 80/20, `random_state=42`.
- **Feature scaling** — `StandardScaler` fit on the training set and
  applied to both splits.

## 🤖 Models & results

Three approaches were built and compared on the same preprocessed features:

| Model | MAE | MSE (Train) | MSE (Test) | R² (Test) |
|---|---|---|---|---|
| Linear Regression | 1.9746 | 5.9732 | 5.2336 | 0.5553 |
| Polynomial Regression (degree 2) | 1.4788 | 2.9884 | 3.0104 | 0.7442 |
| Polynomial + Ridge (α=7) | 1.4594 | 2.9960 | 2.9438 | 0.7498 |
| **Polynomial + Lasso (α=0.1)** | **1.3587** | 3.2895 | **2.4626** | **0.7907** |

Regularization strength was selected by testing multiple alpha values for
both Lasso and Ridge and comparing test-set error.

### Conclusion

- Linear Regression underfit the data — it couldn't capture the non-linear
  fertilizer–yield relationship.
- Polynomial Regression (degree 2) captured the non-linearity and
  substantially improved performance.
- Both Ridge and Lasso regularization improved on the plain polynomial
  model by controlling overfitting; **Lasso (α=0.1) gave the best overall
  test performance** and is the final model used in the app.

## 📁 Project structure

```
.
├── Farm_yield.ipynb          # Full analysis: cleaning, EDA, preprocessing, modeling
├── farm_yield_dataset.csv    # Raw dataset
├── lasso_model.pkl           # Saved model bundle (encoders, scaler, models)
├── farm_yield_app.py         # Interactive Streamlit demo
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

The `lasso_model.pkl` bundle contains:

```python
{
    'Encoders': encoders,               # dict of {column: fitted LabelEncoder}
    'Scaler': scaler,                   # fitted StandardScaler
    'linear_model': linear_model,       # fitted LinearRegression
    'Polynomial Features': poly,        # fitted PolynomialFeatures (degree=2)
    'Lasso': lasso2,                    # fitted Lasso(alpha=0.1) — final model
}
```

## 📦 Requirements

`requirements.txt` covers dependencies for **both** the notebook and the
app — `pandas`, `numpy`, `scikit-learn` are shared; `matplotlib`, `seaborn`,
`jupyter` are for the notebook; `streamlit`, `plotly` are for the app.

```bash
pip install -r requirements.txt
```

## 📓 Explore the analysis

Open `Farm_yield.ipynb` to see the full workflow — outlier handling,
missing-value imputation, EDA plots, correlation analysis, and the model
comparison with and without regularization.

## 🚀 Run the interactive demo

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run farm_yield_app.py
```

The app opens at `http://localhost:8501`. Enter farm conditions in the
sidebar to compare predictions from the Linear and Polynomial+Lasso models
side by side.

## ☁️ Deploy the demo on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo and branch, and set the main file to
   `farm_yield_app.py`.
4. Click **Deploy**.

## 🛠️ Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` / `seaborn` ·
`Streamlit` · `Plotly`
