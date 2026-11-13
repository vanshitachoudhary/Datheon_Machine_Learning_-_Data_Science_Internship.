import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predictive Customer Churn Studio", page_icon="📊", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Predictive Customer Churn & Attrition Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Task 3: Telecom Customer Churn Intelligence Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# Load or generate model pipeline
@st.cache_resource
def load_trained_model():
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20.0, 120.0, n_samples),
        'total_charges': np.random.uniform(50.0, 8000.0, n_samples),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    })
    
    X = df.drop(columns=['churn'])
    y = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    numeric_features = ['tenure', 'monthly_charges', 'total_charges']
    categorical_features = ['contract', 'payment_method']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline

model = load_trained_model()

# Sidebar / Main UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Subscriber Parameters")
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 150.0, 75.5)
    total_charges = st.number_input("Total Charges ($)", 10.0, 10000.0, 950.0)
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    
    predict_btn = st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Attrition Risk Output")
    if predict_btn:
        input_data = pd.DataFrame([{
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract': contract,
            'payment_method': payment_method
        }])
        
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk Detected!**")
            st.metric(label="Calculated Attrition Probability", value=f"{prob:.2f}%")
        else:
            st.success(f"✅ **Customer Retained (Low Risk)**")
            st.metric(label="Churn Probability", value=f"{prob:.2f}%")
    else:
        st.info("👈 Fill in the subscriber details on the left and click **Predict Churn Risk**.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Engineered by Vanshita Choudhary • Streamlit ML Dashboard</p>", unsafe_allow_html=True)
