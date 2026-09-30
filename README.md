# Credit Risk Scorer

Predicts whether a loan applicant will have a serious delinquency (90+ days past due) within two years, using the **Give Me Some Credit** dataset (150,000 borrowers). The project covers the full workflow: data cleaning, exploratory analysis, feature engineering, class balancing with SMOTE, model comparison and SHAP explainability.

## Results

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** (selected) | **0.8301** |
| Random Forest | 0.8128 |
| XGBoost | 0.7795 |

Logistic Regression scored best and is saved as the final model.

### Key findings
- **Imbalanced target:** only 6.6% of applicants defaulted (9,883 of 149,735 after cleaning), so SMOTE was used to balance the training set.
- **Late payments are the strongest signal.** Defaulters averaged 2.1 total late payments, against 0.3 for non-defaulters.
- **Age matters.** Default rates fall with age: 10.9% for applicants under 35, 8.3% for 35–54 and 3.7% for 55+.

## Charts

| | |
|---|---|
| ![Default distribution](images/01_default_distribution.png) | ![Age distribution](images/02_age_distribution.png) |
| ![Income vs default](images/03_income_vs_default.png) | ![Age vs default](images/04_age_vs_default.png) |
| ![Correlation heatmap](images/05_correlation_heatmap.png) | ![Late payments](images/06_late_payments.png) |
| ![SHAP feature importance](images/07_shap_importance.png) | ![SHAP dot plot](images/08_shap_dot.png) |

## Workflow

1. **Load & inspect:** shape, data types, missing values, class balance.
2. **Clean:** drop the index column, remove `age = 0` and the `98` placeholder codes in the late-payment columns, fill missing income and dependents with the median, cap outliers at the 99th percentile.
3. **Explore:** distributions, box plots, correlation heatmap, late payments by default status.
4. **Feature engineering:** `TotalLatePayments`, `IncomePerDependent`, `AgeGroup`.
5. **Model:** 80/20 stratified split, SMOTE on the training set only, `StandardScaler`, then Logistic Regression, Random Forest and XGBoost, compared on ROC-AUC.
6. **Explain:** SHAP summary plots for the chosen model.

## Project structure

```
credit-risk-scorer/
├── data/            # put credit_risk.csv here (not committed)
├── images/          # charts used in this README
├── models/          # saved model and scaler (created when you run the notebook)
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── requirements.txt
└── README.md
```

## How to run

1. Clone the repo and install the dependencies:
   ```bash
   git clone <this-repo-url>
   cd credit-risk-scorer
   pip install -r requirements.txt
   ```
2. Download `cs-training.csv` from the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) competition on Kaggle, rename it to `credit_risk.csv` and place it in `data/`.
3. Open `notebooks/credit_risk_analysis.ipynb` in Jupyter or VS Code and run all cells.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Matplotlib · Seaborn
