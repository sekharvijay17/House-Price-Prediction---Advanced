# 🏠 House Price Prediction: Ames, Iowa (PRCP-1020)

An end-to-end machine learning project that predicts house sale prices, explains which features drive the price, and recommends good-value houses to buyers.

📓 **Notebook:** [House Price Prediction - Advanced.ipynb](https://github.com/sekharvijay17/House-Price-Prediction---Advanced/blob/main/House%20Price%20Prediction%20-%20Advanced.ipynb)

---

## 📌 Project Overview

Using the Ames Housing dataset (**1,460 houses, 79 features**), this project:

1. Analyses the data in depth (Task 1)
2. Builds and compares machine learning models to predict price, and studies how features affect price (Task 2)
3. Recommends houses to buyers by budget, area and other needs (Task 3)
4. Reports model performance and the challenges faced

---

## 📂 Repository Structure

```
├── House Price Prediction - Advanced.ipynb   # Full project notebook
├── data.csv                                  # Ames housing dataset
└── README.md
```

---

## 🧭 Project Workflow

**Load data → Explore → Clean → Engineer features → Split → Train and compare models → Tune → Explain features → Recommend houses → Report**

### Task 1: Data Analysis
- Data types, duplicates and missing values
- SalePrice distribution (right-skewed, so log transform is used)
- Correlation with price, key feature relationships and outliers
- Neighborhood and time-trend analysis

### Data Cleaning
- Missing values filled with `"None"` or `0` where the feature does not exist (for example no pool or no garage)
- `LotFrontage` filled with the neighborhood median, `Electrical` with the mode
- 2 outliers removed (`GrLivArea > 4000` with a low price, from partial sales)
- Quality ratings (Po, Fa, TA, Gd, Ex) converted to numbers 0 to 5

### Feature Engineering
`TotalSF`, `TotalBath`, `TotalPorchSF`, `HouseAge`, `RemodAge`, `IsRemodeled`, `HasPool`, `HasGarage`, `HasBasement`, `Has2ndFloor`, `HasFireplace`, `QualXArea`

### Task 2a: Modeling
- Target: `log1p(SalePrice)`
- Preprocessing `Pipeline`: median imputation and scaling for numbers, most-frequent imputation and one-hot encoding for text
- 80% train / 20% test split, plus 5-fold cross-validation
- Models: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning with `RandomizedSearchCV`

### Task 2b: Feature Relationships
Feature importance, Ridge coefficients, partial dependence plots and a "what-if" price calculator.

### Task 3: Buyer Recommendations
Every house gets an out-of-fold predicted (fair) price. The `ValueScore%` shows how far the actual price is below that fair price. The `recommend_houses()` function filters normal sales by budget, area, bedrooms, bathrooms, quality, garage, age and neighborhood, then ranks them by value.

---

## 📊 Results

| Rank | Model | CV RMSE (log) | Test R² | Test MAE ($) |
|---|---|---|---|---|
| 1 | **XGBoost** | 0.1125 | 0.9232 | 13,211 |
| 2 | ElasticNet | 0.1129 | 0.9190 | 14,046 |
| 3 | Ridge | 0.1132 | 0.9105 | 14,436 |
| 4 | Gradient Boosting | 0.1133 | 0.9198 | 13,689 |
| 5 | Lasso | 0.1139 | 0.9176 | 13,916 |
| 6 | Random Forest | 0.1321 | 0.8836 | 15,877 |
| 7 | Linear Regression | failed | failed | failed |

**Tuned Gradient Boosting (test set):** R² 0.9226 | RMSE $19,533 | MAE $14,162 | MAPE 8.45%

On average, a prediction is off by about **$14,000 (about 8.5% of the price)**.

### Key Takeaways
- XGBoost is the best model. The top five models are very close to each other.
- Regularized linear models (Ridge, Lasso, ElasticNet) are almost as accurate and much simpler.
- Plain Linear Regression failed because of correlated features and 200+ one-hot columns.
- Quality, living area and location are the main drivers of price.

### ✅ Recommendation for Production
Use the **XGBoost pipeline**. **ElasticNet** is a simpler, faster alternative.

---

## ⚠️ Challenges and Solutions

| Challenge | Solution |
|---|---|
| Many missing values | Filled by meaning: "None" or 0 for absent features, neighborhood median for LotFrontage |
| Skewed SalePrice | Log transform |
| Outliers | Removed 2 partial-sale points, kept genuine luxury homes |
| Mixed numeric and text data | `ColumnTransformer` + `OneHotEncoder` in a `Pipeline` |
| Correlated features | Ridge, Lasso, tree models and engineered features |
| Linear Regression instability | Used regularized and tree models |
| Fake bargains in recommendations | Recommend only `SaleCondition == "Normal"` houses |

---

## 🛠️ Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter Notebook

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/sekharvijay17/House-Price-Prediction---Advanced.git
cd House-Price-Prediction---Advanced

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

# 3. Open the notebook (keep data.csv in the same folder)
jupyter notebook "House Price Prediction - Advanced.ipynb"
```

Then choose **Kernel → Restart & Run All**.

---

## 🚀 Future Improvements
- Tune XGBoost and stack models (Ridge + XGBoost + Gradient Boosting)
- Add SHAP explanations for individual predictions
- Deploy as a web app (Streamlit) where users enter house details and get a price

---

## 👤 Author

**sekharvijay17**
GitHub: [@sekharvijay17](https://github.com/sekharvijay17)

---

⭐ If you found this project useful, give it a star!
