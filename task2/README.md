# Task 2: California Housing Price Prediction

Completed for the **Datheon** Machine Learning & Data Science Internship.

OLS vs Ridge (L2) vs Lasso (L1) with GridSearchCV, engineered ratio/interaction features, RMSE / MAE / R2 comparison and residual analysis.

<!-- RESULTS_START -->
## Results

California housing districts: 20,640. Best model by cross-validation: **Lasso (L1)**.

| Model | Test RMSE ($) | Test MAE ($) | Test R2 | 5-fold CV RMSE ($) |
|---|---|---|---|---|
| OLS (all features) | 66,953 | 48,609 | 0.658 | 65,716 |
| OLS (VIF-reduced) | 71,519 | 50,108 | 0.610 | 68,404 |
| Ridge (L2) | 66,968 | 48,618 | 0.658 | 65,704 |
| **Lasso (L1)** | 67,002 | 48,575 | 0.657 | 65,702 |

### Conclusion
- **Best model:** Lasso (L1) (lowest 5-fold CV RMSE, $65,702). On the held-out test set: RMSE $67,002, MAE $48,575, R2 0.657.
- **Regularisation / multicollinearity:** Lasso (L1) lowered CV RMSE from $65,716 (plain OLS) to $65,702 (0.0% lower, a marginal gain).
- **Multicollinearity:** 5 features had VIF > 10 (households, total_bedrooms, total_rooms, median_income, population). OLS on the VIF-reduced set had CV RMSE $68,404 vs $65,716 with all features, so they still carried useful signal; regularisation is the gentler fix.
- **Lasso feature selection:** dropped total_rooms, total_bedrooms, income_x_rooms, ocean_proximity_ISLAND.
- **Strongest drivers (largest standardised Ridge coefficients):** median_income, latitude, longitude.
- **Where it struggles:** largest errors in the Q5 (priciest) price range (MAE $87,120) vs Q4 (MAE $35,258).
- **Next step:** try non-linear models (Random Forest / Gradient Boosting) and location clustering, since linear models cannot capture non-linear effects such as location.

### Plots
![fig1.png](images\fig1.png)

![fig2.png](images\fig2.png)

![fig3.png](images\fig3.png)

![fig4.png](images\fig4.png)

![fig5.png](images\fig5.png)

![fig6.png](images\fig6.png)
<!-- RESULTS_END -->

**Live demo:** <paste your Streamlit link here>

## Dataset
The notebook downloads `housing.csv` automatically on first run from https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv and saves it to `data/housing.csv` (20,640 rows; `total_bedrooms` has nulls, `ocean_proximity` is categorical). You can also place your own copy at `data/housing.csv`. With no internet it falls back to scikit-learn's built-in California Housing.

## Architecture
```
raw data -> schema check -> feature engineering (ratios + interactions) -> train/test split -> [median imputation -> StandardScaler | one-hot encoding] -> VIF multicollinearity check -> OLS / OLS (VIF-reduced) / Ridge / Lasso (GridSearchCV) -> CV comparison -> residual analysis -> saved model -> Streamlit app
```

## Files
```
task2_housing_regression.ipynb   main notebook (run top to bottom)
app.py                 Streamlit demo
models/                saved model (created by the notebook, needed by app.py)
images/                plots saved by the notebook (use in README / LinkedIn)
RESULTS.md             metrics summary written by the notebook
requirements.txt
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
# open task2_housing_regression.ipynb in VS Code (kernel = this .venv) and press Run All, or run it from the terminal:
python -m nbconvert --to notebook --execute --inplace task2_housing_regression.ipynb
streamlit run app.py
```

Internship: [Datheon](https://datheon.in)
