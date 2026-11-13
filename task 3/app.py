from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(__name__)

model_path = 'churn_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Churn model not loaded or trained.'}), 500
    
    data = request.get_json()
    try:
        input_data = pd.DataFrame([{
            'tenure': float(data.get('tenure', 12)),
            'monthly_charges': float(data.get('monthly_charges', 65.0)),
            'total_charges': float(data.get('total_charges', 800.0)),
            'contract': str(data.get('contract', 'Month-to-month')),
            'payment_method': str(data.get('payment_method', 'Electronic check'))
        }])
    except Exception as e:
        return jsonify({'error': f'Invalid input format: {e}'}), 400
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    churn_prob = float(probabilities[1]) * 100
    retain_prob = float(probabilities[0]) * 100
    result_label = "Churn (Attrition Risk)" if prediction == 1 else "Retained (Loyal Customer)"
    
    return jsonify({
        'prediction': result_label,
        'churn_risk_score': round(churn_prob, 2),
        'retain_score': round(retain_prob, 2),
        'is_churn': bool(prediction == 1)
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400
    
    try:
        df = pd.read_csv(file)
        required_cols = ['tenure', 'monthly_charges', 'total_charges', 'contract', 'payment_method']
        for col in required_cols:
            if col not in df.columns:
                match = [c for c in df.columns if c.lower() == col]
                if match:
                    df.rename(columns={match[0]: col}, inplace=True)
                else:
                    df[col] = 12 if col == 'tenure' else (65.0 if 'charges' in col else 'Month-to-month')
        
        X_batch = df[required_cols]
        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]
        
        df['Predicted_Status'] = ['Churn Risk' if p == 1 else 'Retained' for p in preds]
        df['Churn_Probability_Pct'] = (probs * 100).round(2)
        
        return jsonify({
            'success': True,
            'total_analyzed': len(df),
            'preview': df.head(10).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('churn_model.pkl'):
        from train_model import train_and_save
        train_and_save()
    app.run(debug=True, port=5000)
