import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Spam Email Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Blue and White Theme & Modern Design
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0D47A1 0%, #1E88E5 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(13, 71, 161, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
    }
    .hero-banner p {
        font-size: 1.15rem;
        opacity: 0.92;
        max-width: 750px;
        margin: 0 auto;
        color: #E3F2FD !important;
    }
    
    /* Cards and Containers */
    .custom-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    
    /* Streamlit Text Area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        font-size: 1rem;
        padding: 0.75rem;
    }
    .stTextArea textarea:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.2);
    }
    
    /* Result Badges */
    .spam-badge {
        background: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .spam-badge h2 {
        color: #DC2626 !important;
        font-weight: 800;
        margin: 0;
        font-size: 1.8rem;
    }
    
    .ham-badge {
        background: #F0FDF4;
        border: 2px solid #22C55E;
        color: #166534;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .ham-badge h2 {
        color: #16A34A !important;
        font-weight: 800;
        margin: 0;
        font-size: 1.8rem;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    /* Sample Chips */
    .sample-chip-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    
    /* Stat Cards */
    .stat-box {
        background-color: #F1F5F9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0D47A1;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Hide Streamlit default menu padding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Artifacts Safely
# ---------------------------------------------------------
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        try:
            metrics = joblib.load("metrics_summary.pkl")
        except Exception:
            metrics = None
        return model, vectorizer, metrics
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure 'train_model.py' has executed successfully.")
        return None, None, None

model, vectorizer, metrics = load_ml_components()

# ---------------------------------------------------------
# NLP Text Preprocessing Pipeline
# ---------------------------------------------------------
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
    if not text or not isinstance(text, str):
        return ""
    text_lower = text.lower()
    text_no_url = re.sub(r'https?://\S+|www\.\S+', '', text_lower)
    text_no_punct = text_no_url.translate(str.maketrans('', '', string.punctuation))
    text_no_digits = re.sub(r'\d+', '', text_no_punct)
    words = text_no_digits.split()
    cleaned_tokens = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(cleaned_tokens)

# ---------------------------------------------------------
# Sample Email Data
# ---------------------------------------------------------
SAMPLE_EMAILS = {
    "🎁 $1000 Gift Card Scam (Spam)": "URGENT! You have won a $1,000 Walmart Gift Card. Claim your reward immediately at http://bit.ly/claim-reward-now before it expires in 2 hours!",
    "⚠️ Bank Phishing Alert (Spam)": "IMPORTANT NOTICE: Your Bank Account has been compromised due to suspicious logins. Verify your credentials now at https://secure-bank-login-verify.com to avoid permanent account freeze.",
    "💼 Quarterly Report Sync (Not Spam)": "Hi Alex, please find attached the quarterly financial report for Q3. Let me know if you have any questions before our 2 PM team meeting.",
    "📦 Order Delivery Notice (Not Spam)": "Your order #94821 has been shipped via UPS. Expected delivery date is Thursday, Aug 8th. Track your package in the customer portal."
}

# Session state initialization
if "email_input" not in st.session_state:
    st.session_state.email_input = ""

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield-with-signature.png", width=70)
    st.title("Navigation")
    app_mode = st.radio("Go to:", ["📩 Detector", "📊 Model Metrics", "ℹ️ About Project", "👥 Team Members"])
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if model is not None and vectorizer is not None:
        st.success("🟢 ML Engine Loaded")
        if metrics:
            st.info(f"🏆 Best Model: **{metrics['best_model_name']}**")
    else:
        st.error("🔴 Model Not Found")
    
    st.markdown("---")
    st.caption("AI Spam Detection System v1.0 • Built with Streamlit & Scikit-Learn")

# ---------------------------------------------------------
# Main Page Render: 📩 Detector
# ---------------------------------------------------------
if app_mode == "📩 Detector":
    # Hero Header Banner
    st.markdown("""
        <div class="hero-banner">
            <h1>🛡️ AI Spam Email Detection</h1>
            <p>Empowered by Natural Language Processing and Machine Learning to classify emails in real-time with high precision and transparency.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_input, col_preset = st.columns([7, 4], gap="medium")
    
    with col_input:
        st.markdown("### ✉️ Email Text Input")
        st.caption("Paste or type the contents of an email below to analyze its legitimacy.")
        
        email_text = st.text_area(
            label="Email Content",
            value=st.session_state.email_input,
            height=240,
            placeholder="Type or paste the email subject & body text here...",
            key="text_area_email"
        )
        
        btn_col1, btn_col2, btn_col3 = st.columns([3, 3, 4])
        with btn_col1:
            detect_clicked = st.button("🔍 Detect Spam", type="primary", use_container_width=True)
        with btn_col2:
            clear_clicked = st.button("🗑️ Clear Input", use_container_width=True)
        
        if clear_clicked:
            st.session_state.email_input = ""
            st.rerun()
            
    with col_preset:
        st.markdown("### ⚡ Quick Test Samples")
        st.caption("Click any sample to load it directly into the analyzer:")
        
        for name, text in SAMPLE_EMAILS.items():
            if st.button(name, use_container_width=True):
                st.session_state.email_input = text
                st.rerun()

    # Classification & Results Logic
    current_input = email_text.strip()
    
    if detect_clicked or (current_input and "auto_run" in st.session_state and st.session_state.auto_run):
        if not current_input:
            st.warning("⚠️ Please enter or paste an email message before clicking 'Detect Spam'.")
        elif model is None or vectorizer is None:
            st.error("❌ Machine Learning model artifacts are missing. Run `train_model.py` first.")
        else:
            with st.spinner("🧠 Processing NLP pipeline and running ML prediction..."):
                # Preprocess input text
                clean_input = preprocess_text(current_input)
                
                # Transform via TF-IDF
                vectorized_input = vectorizer.transform([clean_input])
                
                # Predict
                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                
                ham_prob = round(probabilities[0] * 100, 2)
                spam_prob = round(probabilities[1] * 100, 2)
                
                is_spam = (prediction == 1)
                confidence = spam_prob if is_spam else ham_prob
                
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            res_col1, res_col2 = st.columns([5, 6], gap="large")
            
            with res_col1:
                if is_spam:
                    st.markdown(f"""
                        <div class="spam-badge">
                            <h2>🚨 SPAM DETECTED</h2>
                            <p style="margin-top: 8px; font-weight: 600; font-size: 1.1rem;">
                                Danger level high: Potential phishing or scam email.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="ham-badge">
                            <h2>✅ NOT SPAM (HAM)</h2>
                            <p style="margin-top: 8px; font-weight: 600; font-size: 1.1rem;">
                                Safe email: Appears to be authentic communication.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 🎯 Prediction Metrics")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Classification", "SPAM" if is_spam else "NOT SPAM")
                with m2:
                    st.metric("Confidence Score", f"{confidence:.1f}%")
            
            with res_col2:
                st.markdown("#### ⚖️ Prediction Probabilities")
                
                st.write(f"**Spam Probability:** {spam_prob}%")
                st.progress(float(spam_prob / 100))
                
                st.write(f"**Legitimate (Ham) Probability:** {ham_prob}%")
                st.progress(float(ham_prob / 100))
                
                with st.expander("🔍 Inspect Processed NLP Tokens", expanded=True):
                    st.markdown("**Original Character Length:** " + str(len(current_input)))
                    st.markdown("**Processed Clean Text (Lowercase, Stemmed, No Stopwords/Punctuation):**")
                    st.code(clean_input if clean_input else "[No tokens remaining after filtering]")

# ---------------------------------------------------------
# Page Render: 📊 Model Metrics
# ---------------------------------------------------------
elif app_mode == "📊 Model Metrics":
    st.markdown("""
        <div class="hero-banner">
            <h1>📊 Model Evaluation & Comparison</h1>
            <p>Benchmark results of Naive Bayes, Logistic Regression, and Random Forest on the evaluation dataset.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if metrics and "results" in metrics:
        res = metrics["results"]
        df_metrics = pd.DataFrame(res).T
        
        st.markdown("### 🏆 Algorithm Performance Overview")
        
        cols = st.columns(len(res))
        for idx, (m_name, m_scores) in enumerate(res.items()):
            with cols[idx]:
                st.markdown(f"""
                    <div class="custom-card" style="text-align: center;">
                        <h3 style="color: #0D47A1; margin-bottom: 0.5rem;">{m_name}</h3>
                        <div class="stat-number">{m_scores['Accuracy']}%</div>
                        <div class="stat-label">Accuracy</div>
                        <hr style="margin: 0.8rem 0;">
                        <p style="font-size: 0.9rem; color: #475569; margin: 0;">
                            Precision: <b>{m_scores['Precision']}%</b><br>
                            Recall: <b>{m_scores['Recall']}%</b><br>
                            F1-Score: <b>{m_scores['F1-Score']}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown("### 📋 Detailed Comparison Table")
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#BBDEFB"), use_container_width=True)
        
        st.markdown("### 💡 Why Naive Bayes / Selected Model Wins?")
        st.write("""
        - **Multinomial Naive Bayes** excels at text classification because word probabilities are conditionally independent given the class, making it extremely fast, sample-efficient, and effective with TF-IDF features.
        - **Logistic Regression** provides a robust linear baseline with smooth probability estimates.
        - **Random Forest** captures non-linear feature combinations but requires more depth for sparse text matrices.
        """)
    else:
        st.warning("No saved metrics summary found. Please run `train_model.py` to generate complete evaluation metrics.")

# ---------------------------------------------------------
# Page Render: ℹ️ About Project
# ---------------------------------------------------------
elif app_mode == "ℹ️ About Project":
    st.markdown("""
        <div class="hero-banner">
            <h1>ℹ️ About AI Spam Email Detection</h1>
            <p>Learn about the architecture, Natural Language Processing pipeline, and machine learning methodology.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Technical Architecture
    This application utilizes an end-to-end Machine Learning and NLP pipeline designed for high accuracy and fast inference:
    
    1. **Text Normalization & Preprocessing**:
       - **Lowercasing**: Standardizes all incoming text to lower case.
       - **Noise & Punctuation Removal**: Strips HTML tags, URLs, numbers, and special symbols using Regular Expressions.
       - **Stopwords Filtering**: Filters out uninformative English stop words (e.g. *the, is, at, which*).
       - **Porter Stemming**: Reduces words to their morphological root stem (e.g., *claiming* -> *claim*).
    
    2. **Feature Extraction**:
       - **TF-IDF (Term Frequency - Inverse Document Frequency)** vectorization converts text tokens into unigram and bigram numerical vectors, measuring word importance across the corpus.
    
    3. **Machine Learning Classification**:
       - Evaluates **Multinomial Naive Bayes**, **Logistic Regression**, and **Random Forest**.
       - Automatically saves and loads the best-performing model (`model.pkl`) and fitted vectorizer (`vectorizer.pkl`).
    
    4. **Streamlit Interactive UI**:
       - High-speed interactive web interface featuring custom CSS, instant predictions, sample presets, and detailed confidence analytics.
    """)

# ---------------------------------------------------------
# Page Render: 👥 Team Members
# ---------------------------------------------------------
elif app_mode == "👥 Team Members":
    st.markdown("""
        <div class="hero-banner">
            <h1>👥 Project Team & Contributors</h1>
            <p>Meet the engineers and data scientists behind the AI Spam Detection Web Application.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="custom-card" style="text-align: center;">
                <img src="https://img.icons8.com/color/96/000000/user-female-circle.png" width="80">
                <h3 style="margin-top: 10px; color: #0D47A1;">Lead AI Developer</h3>
                <p style="color: #64748B; font-weight: 500;">NLP & Machine Learning Specialist</p>
                <p style="font-size: 0.9rem; color: #475569;">Designed the preprocessing pipeline, model training, and automated model selection workflow.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="custom-card" style="text-align: center;">
                <img src="https://img.icons8.com/color/96/000000/administrator-male.png" width="80">
                <h3 style="margin-top: 10px; color: #0D47A1;">Full Stack UI Engineer</h3>
                <p style="color: #64748B; font-weight: 500;">Streamlit & Frontend Architect</p>
                <p style="font-size: 0.9rem; color: #475569;">Crafted the blue-and-white theme, sample presets, clear button state, and probability visualizations.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="custom-card" style="text-align: center;">
                <img src="https://img.icons8.com/color/96/000000/conference-call.png" width="80">
                <h3 style="margin-top: 10px; color: #0D47A1;">QA & MLOps Lead</h3>
                <p style="color: #64748B; font-weight: 500;">Model Validation & Deployment</p>
                <p style="font-size: 0.9rem; color: #475569;">Handled dataset curation, metric evaluation, model pickling, and production deployment setup.</p>
            </div>
        """, unsafe_allow_html=True)
