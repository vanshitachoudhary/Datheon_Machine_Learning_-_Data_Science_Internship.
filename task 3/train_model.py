import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

def train_and_save():
    try:
        df = pd.read_csv('telco_churn_data.csv')
        print("Successfully loaded telco_churn_data.csv dataset.")
    except Exception as e:
        print(f"Could not load CSV: {e}. Generating fallback data.")
        df = pd.DataFrame({
            'tenure': [12, 34, 5, 48, 2],
            'monthly_charges': [70.5, 25.0, 90.0, 45.5, 95.0],
            'total_charges': [850.0, 900.0, 450.0, 2200.0, 190.0],
            'contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month'],
            'payment_method': ['Electronic check', 'Bank transfer', 'Electronic check', 'Credit card', 'Mailed check'],
            'churn': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
    
    df.columns = [col.strip().lower() for col in df.columns]
    X = df.drop(columns=['churn', 'customer_id'], errors='ignore')
    y = df['churn'].apply(lambda x: 1 if str(x).strip().capitalize() == 'Yes' else 0)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    print(f"Model Training Complete | Accuracy: {acc*100:.2f}% | ROC-AUC Score: {roc_auc*100:.2f}%")
    
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved successfully as churn_model.pkl")

if __name__ == '__main__':
    train_and_save()
