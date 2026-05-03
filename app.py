import streamlit as st
from transformers import pipeline
import torch
import os
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 1.1rem !important; color: #1e1e1e; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING WITH ERROR CHECKING ---
@st.cache_resource
def load_classifier():
    model_path = Path("spam_model")
    
    # 1. Check if folder exists
    if not model_path.exists():
        return None, f"Folder '{model_path}' not found. Please check GitHub structure."
    
    # 2. Check for essential files
    required_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    missing = [f for f in required_files if not (model_path / f).exists()]
    if missing:
        return None, f"Missing files in model folder: {', '.join(missing)}"

    try:
        # 3. Load the pipeline
        pipe = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=-1  # Force CPU for Streamlit Cloud stability
        )
        return pipe, "Success"
    except Exception as e:
        return None, f"Pipeline Error: {str(e)}"

# Initialize model
classifier, model_status = load_classifier()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ System Control")
    if classifier:
        st.success("✅ Model Status: Ready")
    else:
        st.error(f"❌ Model Status: {model_status}")
    
    st.divider()
    st.subheader("Model Calibration")
    # This allows you to toggle between labels if your model is inverted
    spam_label = st.selectbox(
        "Select the label that represents SPAM:",
        options=["LABEL_1", "LABEL_0", "SPAM", "1"],
        index=0,
        help="Check 'Technical Metadata' after a scan to see which label your model uses for spam."
    )
    
    st.divider()
    st.caption("Muhammad Hamza • Qassim University • MS in AI")

# --- MAIN INTERFACE ---
st.title("🛡️ SpamGuard AI")
st.markdown("### Intelligent Spam & Phishing Detection")
st.write("Paste the message content below to analyze it using our fine-tuned DistilBERT model.")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Enter Message:",
        height=250,
        placeholder="e.g., 'Congratulations! You've won a $1000 gift card...'",
    )
    
    analyze_btn = st.button("🔍 Analyze Message", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter a message to scan.")
        elif classifier is None:
            st.error("The AI model is not loaded. Check the sidebar for details.")
        else:
            with st.spinner("AI is evaluating message patterns..."):
                try:
                    # Run prediction
                    result = classifier(user_input[:512])[0]
                    label = result['label'].upper()
                    score = result['score']
                    
                    # Logic Check
                    is_spam = (label == spam_label.upper())
                    
                    st.divider()
                    
                    if is_spam:
                        st.error("### 🚨 HIGH RISK: SPAM", icon="⚠️")
                        st.metric("Spam Confidence", f"{score:.1%}")
                        st.warning("**Recommendation:** This message shows signs of phishing. Do not click links.")
                    else:
                        st.success("### ✅ LOW RISK: SAFE", icon="✅")
                        st.metric("Safety Confidence", f"{score:.1%}")
                        st.info("**Analysis:** This message appears to be legitimate.")

                    # Technical Debugging
                    with st.expander("Technical Metadata"):
                        st.write(f"Detected Label: `{label}`")
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    else:
        st.info("Input a message and click 'Analyze' to see the security score.")

st.divider()
