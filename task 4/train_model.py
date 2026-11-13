import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_and_save():
    try:
        df = pd.read_csv('sentiment_reviews.csv', quotechar='"', engine='python')
        print("Successfully loaded sentiment_reviews.csv dataset.")
    except Exception as e:
        print(f"Could not load CSV file: {e}. Using fallback data.")
        df = pd.DataFrame({
            'review_text': [
                "The product quality is exceptional, exceeded all my expectations!",
                "Terrible customer service and the item arrived completely damaged.",
                "Average performance, decent for the price point but nothing special."
            ],
            'sentiment': ['positive', 'negative', 'neutral']
        })
    
    df.columns = [col.strip().lower() for col in df.columns]
    text_col = 'review_text' if 'review_text' in df.columns else df.columns[1]
    target_col = 'sentiment' if 'sentiment' in df.columns else df.columns[2]
    
    X = df[text_col].astype(str)
    y = df[target_col].astype(str).str.capitalize()
    
    # Build ML Pipeline (TF-IDF + Logistic Regression)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Model Training Complete. Training Accuracy: {acc * 100:.2f}%")
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved successfully as model.pkl")

if __name__ == '__main__':
    train_and_save()
