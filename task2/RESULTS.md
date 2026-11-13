## Test-set results

```
             Model   RMSE ($)    MAE ($)     R2 Best alpha
OLS (all features) 66953.0638 48609.3139 0.6579          -
 OLS (VIF-reduced) 71518.5849 50107.7947 0.6097          -
        Ridge (L2) 66968.1333 48618.2677 0.6578     3.1623
        Lasso (L1) 67002.2667 48574.7524 0.6574      0.001
```

## 5-fold CV RMSE (training set)

```
             Model  CV RMSE ($) mean  CV RMSE ($) std
OLS (all features)           65715.6           1373.2
 OLS (VIF-reduced)           68404.4           1483.9
        Ridge (L2)           65703.5           1352.5
        Lasso (L1)           65701.9           1372.7
```

Best model (by CV): Lasso (L1)
VIF > 10 features: ['households', 'total_bedrooms', 'total_rooms', 'median_income', 'population']
Dropped in VIF-reduced model: ['total_bedrooms', 'households', 'median_income']
