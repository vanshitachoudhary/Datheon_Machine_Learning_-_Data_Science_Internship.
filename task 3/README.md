# Task 3: Predictive Customer Churn Classifier & Intelligence Studio

An enterprise-grade Machine Learning web application to predict customer churn using telecom subscription records, addressing severe class imbalance with robust preprocessing, Random Forest & XGBoost classifiers, real-time single-customer churn inference, bulk CSV batch scoring, and a modern glassmorphic Tailwind CSS dashboard UI.

## Project Structure
```text
task 3/
│
├── app.py                 # Flask backend REST API & prediction routes
├── train_model.py         # ML training pipeline (Scaling, OneHotEncoder, Random Forest, ROC-AUC)
├── telco_churn_data.csv   # Telecom subscriber training & evaluation dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── templates/
    └── index.html         # Glassmorphic frontend dashboard (Tailwind CSS)
```

## Tech Stack
- **Backend:** Python, Flask, Scikit-Learn, Pandas, NumPy, Pickle
- **Frontend:** HTML5, Tailwind CSS, JavaScript (Fetch API, FontAwesome)
- **Model:** ColumnTransformer Pipeline + Random Forest Classifier (`class_weight='balanced'`)

## Quick Start & Installation
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model using the dataset:
   ```bash
   python train_model.py
   ```
3. Run the Flask web application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to: **`http://127.0.0.1:5000`**
