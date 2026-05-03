import streamlit as st
from transformers import pipeline
import torch
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTextArea textarea { font-size: 1.1rem !important; }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_classifier():
    model_path = Path("spam_model")
    try:
        # We use the standard pipeline but specify the local path
        pipe = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=-1  # Use CPU for deployment stability
        )
        return pipe
    except Exception as e:
        st.error(f"Could not load model weights: {e}")
        return None

classifier = load_classifier()

# --- SIDEBAR / SETTINGS ---
with st.sidebar:
    st.title("⚙️ Model Settings")
    st.info("This app uses a fine-tuned DistilBERT model to detect phishing and spam.")
    
    # This is the fix for your "LABEL_0" issue
    # If your model trained LABEL_0 as Spam, toggle this OFF.
    spam_label = st.selectbox("Which label is SPAM?", ["LABEL_1", "LABEL_0", "SPAM"], index=0)
    
    st.divider()
    st.caption("Developed by Muhammad Hamza\nQassim University • MS in AI")

# --- MAIN INTERFACE ---
st.title("🛡️ SpamGuard AI")
st.write("Enter the content of an email or SMS below to analyze it for security risks.")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Message Content:",
        height=250,
        placeholder="Paste your email or SMS here...",
        help="The AI performs best with English text under 512 words."
    )
    
    analyze_btn = st.button("🔍 Run Security Scan", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    if analyze_btn and user_input.strip():
        with st.spinner("Analyzing patterns..."):
            # Execute prediction
            prediction = classifier(user_input[:512])[0]
            label = prediction['label']
            score = prediction['score']
            
            # Classification Logic
            is_spam = (label == spam_label)
            
            if is_spam:
                st.error("### 🚨 SPAM DETECTED")
                st.progress(score, text=f"Spam Probability: {score:.1%}")
                st.warning("**Recommendation:** Do not click links or provide personal data.")
            else:
                st.success("### ✅ MESSAGE SAFE")
                st.progress(score, text=f"Safety Confidence: {score:.1%}")
                st.info("**Analysis:** No common phishing patterns detected.")

            # Research/Debug Data
            with st.expander("View AI Metadata"):
                st.write(f"Raw Label: `{label}`")
                st.write(f"Confidence: `{score:.4f}`")
                st.json(prediction)
    
    elif analyze_btn:
        st.warning("Please enter text to analyze.")
    else:
        st.write("Results will appear here after scanning.")

# --- FOOTER ---
st.divider()
st.markdown("<center>Focused on Traffic Safety & Cybersecurity through Artificial Intelligence</center>", unsafe_allow_html=True)
