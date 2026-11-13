# NLP Sentiment Classification Studio (Task 4)

An enterprise-grade, production-ready Natural Language Processing (NLP) web application for customer review sentiment analysis, featuring TF-IDF feature extraction, Logistic Regression modeling, real-time single-review inference, bulk CSV batch processing, probability breakdown, and a modern glassmorphic dashboard UI.

## Project Structure
```text
NLP_Sentiment_Classification_Studio/
│
├── app.py                 # Flask backend REST API & routes
├── train_model.py         # ML training pipeline (TF-IDF + Logistic Regression)
├── sentiment_reviews.csv  # Customer feedback training/evaluation dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── templates/
    └── index.html         # Glassmorphic frontend dashboard (Tailwind CSS)
```

## Tech Stack
- **Backend:** Python, Flask, Scikit-Learn, Pandas, NumPy, Pickle
- **Frontend:** HTML5, Tailwind CSS, JavaScript (Fetch API, FontAwesome)
- **Model:** TF-IDF Vectorizer (`ngram_range=(1, 2)`) + Logistic Regression Classifier

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
