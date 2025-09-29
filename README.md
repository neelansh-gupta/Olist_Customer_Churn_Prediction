# Olist Customer Churn Prediction

Project: Customer churn prediction using RFM features (Recency, Frequency, Monetary) built from the Olist Brazilian e‑commerce dataset. The notebook engineers behavioral features per customer and predicts a binary churn label (churn if no purchase in last 180 days).

## Data Source
- olist_churn_prediction.ipynb — Main analysis notebook (RFM feature engineering, EDA, modeling).
Dataset - https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Requirements
- Python 3.9+
- Packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to run
1. Use only these Olist CSVs(from the dataset link):
    olist_customers_dataset.csv, 
    olist_orders_dataset.csv, 
    olist_order_payments_dataset.csv
2. Place the three CSV files in the same folder as the notebook.
3. Open `olist_churn_prediction.ipynb` in JupyterLab/Notebook or VS Code.
4. Run cells sequentially. The notebook:
   - Loads and validates the CSVs
   - Filters delivered orders
   - Aggregates payments and computes RFM per customer
   - Labels churn with a 180-day threshold
   - Trains and evaluates Logistic Regression and Random Forest models
   - Saves EDA and model output images to disk

## Expected outputs
- Console summary of dataset, features, model metrics.
- Plots saved: `rfm_analysis.png`, `confusion_matrix_best_model.png`, (`feature_importance.png` if available).

## Reproducibility
- Random seed: 42 (set in notebook)
- Train/test split: 80/20 with stratification
- Cross-validation: 5-fold stratified (as implemented)

## Notes & next steps
- Churn definition is fixed at 180 days; consider testing 90/120/270 days.
- Add richer features (product categories, session behavior, seasonality).
- Explore time-aware validation and advanced models (XGBoost, LightGBM).
- Consider threshold optimization based on business costs.

## License & attribution
Data: Olist (Kaggle). This repository contains analysis code and derived outputs for exploratory and educational purposes.
