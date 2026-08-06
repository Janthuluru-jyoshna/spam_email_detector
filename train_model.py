import re
import string
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources safely
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Initialize Stemmer and Stopwords
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                  "he", "him", "his", "she", "her", "it", "its", "they", "them", "what", "which", "who", 
                  "this", "that", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", 
                  "had", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", 
                  "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", 
                  "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", 
                  "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"}

def preprocess_text(text):
    """
    Standardized NLP text preprocessing pipeline:
    1. Convert text to lowercase
    2. Remove URLs, punctuation & special characters
    3. Remove digits/numbers
    4. Tokenize into individual words
    5. Remove English stopwords
    6. Apply Stemming (PorterStemmer)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 4. Tokenization & 5. Remove Stopwords & 6. Stemming
    words = text.split()
    cleaned_tokens = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 1]
    
    return " ".join(cleaned_tokens)

def train_and_evaluate():
    print("=" * 60)
    print(" STARTING ML SPAM DETECTION TRAINING PIPELINE ")
    print("=" * 60)
    
    # Load Dataset
    csv_path = "spam.csv"
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure correct column naming
    if 'Category' in df.columns and 'Message' in df.columns:
        df = df.rename(columns={'Category': 'label', 'Message': 'text'})
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    
    # Binary mapping: spam=1, ham=0
    df['target'] = df['label'].map({'spam': 1, 'ham': 0})
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Label Distribution:\n{df['label'].value_counts()}\n")
    
    # Preprocess text
    print("Preprocessing text data (lowercasing, punctuation removal, digits removal, stopword removal, stemming)...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )
    
    # Feature Extraction with TF-IDF Vectorizer
    print("Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Models to train
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_obj = None
    
    print("\n" + "=" * 60)
    print(" TRAINING AND EVALUATING CANDIDATE MODELS ")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "model_obj": model
        }
        
        print(f"\n- {name}:")
        print(f"   Accuracy:  {acc * 100:.2f}%")
        print(f"   Precision: {prec * 100:.2f}%")
        print(f"   Recall:    {rec * 100:.2f}%")
        print(f"   F1-Score:  {f1 * 100:.2f}%")
        
        # Select best model based on F1-score & Accuracy
        if f1 > best_f1_score or (f1 == best_f1_score and acc > results.get(best_model_name, {}).get("Accuracy", 0)):
            best_f1_score = f1
            best_model_name = name
            best_model_obj = model
            
    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name} (F1-Score: {best_f1_score * 100:.2f}%)")
    print("=" * 60)
    
    # Save artifacts
    joblib.dump(best_model_obj, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    
    # Save metrics metadata for Streamlit display
    metrics_summary = {
        "best_model_name": best_model_name,
        "results": {
            k: {m: round(v[m] * 100, 2) for m in ["Accuracy", "Precision", "Recall", "F1-Score"]}
            for k, v in results.items()
        }
    }
    joblib.dump(metrics_summary, "metrics_summary.pkl")
    
    print("\nSuccessfully saved model.pkl, vectorizer.pkl, and metrics_summary.pkl!")

if __name__ == "__main__":
    train_and_evaluate()
