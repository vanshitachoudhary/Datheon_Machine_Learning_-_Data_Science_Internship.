from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(__name__)

model_path = 'model.pkl'
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
        return jsonify({'error': 'Model not trained or loaded.'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Please enter valid review text.'}), 400
    
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    
    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities)) * 100

    return jsonify({
        'sentiment': str(prediction),
        'confidence': round(confidence, 2),
        'probabilities': prob_dict
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
        text_col = None
        for col in df.columns:
            if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
                text_col = col
                break
        
        if not text_col:
            text_col = df.columns[0]
            
        reviews = df[text_col].astype(str).tolist()
        predictions = model.predict(reviews)
        
        df['Predicted_Sentiment'] = predictions
        counts = pd.Series(predictions).value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'total_analyzed': len(reviews),
            'distribution': counts,
            'preview': df.head(10).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('model.pkl'):
        from train_model import train_and_save
        train_and_save()
    app.run(debug=True, port=5000)
