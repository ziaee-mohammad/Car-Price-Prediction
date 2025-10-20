# 🚗 Car Price Prediction

An end‑to‑end **regression** project to predict **used car prices** from tabular attributes (make/model, year, mileage, engine specs, fuel type, transmission, etc.).  
Includes clean preprocessing pipelines, feature engineering, model training (Linear Regression, Random Forest, XGBoost), and rigorous evaluation (**MAE, RMSE, R², MAPE**).

---

## 📖 Overview
This repository demonstrates best practices for **tabular regression**:
- Leakage‑safe pipelines with `ColumnTransformer` + `Pipeline`
- Robust handling of categorical & numerical features
- Cross‑validated model selection and error analysis
- Exportable artifacts for inference

---

## 🗂️ Dataset
Typical CSV layout under `data/`:
```
data/
├─ train.csv
└─ test.csv
```
**Columns (examples):** `price`, `make`, `model`, `year`, `mileage`, `engine`, `power`, `fuel`, `transmission`, `owner`, `city`, `features...`

> Ensure target column is named `price` (or update code accordingly). Clean outliers and inconsistent units (e.g., mileage km vs mi).

---

## 🔧 Preprocessing & Feature Engineering
- **Imputation**: numeric (median), categorical (most frequent)
- **Scaling**: Standardize numerical features
- **Encoding**: One‑hot for categoricals (`handle_unknown="ignore"`)
- **Domain features**: car **age** (`current_year - year`), log‑price, mileage buckets
- **Optional**: Target encoding (inside CV), interaction terms

All steps live **inside** `Pipeline` to avoid leakage.

---

## 🧠 Models
- **Linear Regression / Ridge / Lasso** — fast baselines
- **Random Forest Regressor** — non‑linear baseline
- **XGBoost Regressor** — strong tabular learner
- *(Optional)* LightGBM / CatBoost

---

## 📈 Evaluation
Primary metrics:
- **MAE** — median error magnitude
- **RMSE** — penalizes large errors
- **R²** — explained variance
- **MAPE** — relative error (watch for zero/near‑zero prices)

**Error analysis**:
- Residual plots vs. mileage/age
- Feature importance (tree‑based models)
- Price distribution before/after cleaning

---

## 🧩 Repository Structure (suggested)
```
Car-Price-Prediction/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_training_baselines.ipynb
│  ├─ 03_error_analysis.ipynb
├─ src/
│  ├─ data.py          # loading/splitting; type casting
│  ├─ features.py      # preprocessors & domain features
│  ├─ models.py        # model builders (linreg/RF/XGB)
│  ├─ train.py         # CV + fit
│  ├─ eval.py          # metrics & plots
│  └─ infer.py         # single‑row inference
├─ reports/figures/    # residuals, importances, distributions
├─ data/               # (gitignored)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Setup & Usage
1) **Clone & install**
```bash
git clone https://github.com/ziaee-mohammad/Car-Price-Prediction.git
cd Car-Price-Prediction
pip install -r requirements.txt
```

2) **Run notebooks**
```bash
jupyter notebook
```

3) **(Optional) Scripts**
```bash
python -m src.train --model xgboost
python -m src.eval  --report
python -m src.infer --json '{"make":"Toyota","model":"Corolla","year":2018,"mileage":65000,"fuel":"Petrol","transmission":"Manual"}'
```

---

## 📦 Requirements (example)
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

---

## ✅ Good Practices
- Remove duplicates & extreme outliers (e.g., price < 500 or > 200,000 as per market)
- Normalize free‑text fields (model trims/variants)
- Use consistent currency; document exchange rate if converted
- Log‑transform price if heavy‑tailed, and inverse‑transform predictions

---

## 🏷 Tags
```
data-science
machine-learning
regression
tabular-data
feature-engineering
model-evaluation
xgboost
python
jupyter-notebook
```

---

## 👤 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad
👉 Instagram: [@ziaee_mohammad](https://www.instagram.com/ziaee_mohammad/)
---

## 📜 License
MIT — free to use and adapt with attribution.
