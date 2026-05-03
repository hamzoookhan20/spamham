import streamlit as st
from transformers import pipeline
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

# Custom Professional UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 1.1rem !important; border-radius: 10px; }
    .stButton>button { border-radius: 20px; height: 3em; width: 100%; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SpamGuard AI")
st.markdown("### Neural Spam Detection System")
st.caption("Researcher: Muhammad Hamza • Qassim University • MS in AI")
st.divider()

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_spam_model():
    # Ensure your folder on GitHub is named exactly 'spam_model'
    model_path = "spam_model" 
    
    if not os.path.exists(model_path):
        return None, "Folder 'spam_model' not found."
    
    try:
        # Loading the local fine-tuned weights
        pipe = pipeline(
            "text-classification", 
            model=model_path, 
            tokenizer=model_path, 
            device=-1
        )
        return pipe, "Success"
    except Exception as e:
        return None, str(e)

classifier, status_msg = load_spam_model()

# --- 3. USER INTERFACE ---
user_input = st.text_area(
    "Analyze Message Content:", 
    height=200, 
    placeholder="Paste email, SMS, or chat text here to check for phishing patterns..."
)

if st.button("🚀 Analyze Now", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif classifier is None:
        st.error(f"Model Error: {status_msg}")
    else:
        with st.spinner("Running deep learning inference..."):
            # Get model prediction
            result = classifier(user_input[:512])[0]
            label = result['label']
            score = result['score']
            
            # --- 4. FIXED LOGIC (Matching your LABEL_0 result) ---
            # We check for LABEL_0 because your previous test confirmed it as the spam trigger.
            is_spam = (label == "LABEL_0")
            
            st.subheader("Classification Result")
            
            if is_spam:
                st.error("### 🚨 HIGH RISK: SPAM DETECTED")
                col1, col2 = st.columns(2)
                col1.metric("Detection Confidence", f"{score:.1%}")
                col2.metric("Status", "Flagged")
                st.write("🔍 **Analysis:** This message matches linguistic patterns found in phishing and fraudulent emails.")
            else:
                st.success("### ✅ LOW RISK: SAFE MESSAGE")
                col1, col2 = st.columns(2)
                col1.metric("Safety Confidence", f"{score:.1%}")
                col2.metric("Status", "Verified")
                st.write("🔍 **Analysis:** No malicious patterns detected. The message appears to be legitimate.")

            # --- 5. RESEARCHER DATA (Technical View) ---
            with st.expander("📊 Technical Metadata (For MS-AI Research)"):
                st.json({
                    "model_output": label,
                    "confidence_score": score,
                    "model_path": "Local /spam_model/",
                    "logic_applied": "LABEL_0 == SPAM"
                })
                if score < 0.60:
                    st.warning("⚠️ **Note:** The model has low confidence in this result. Consider retrying with more text.")

st.divider()
st.caption("© 2026 AI Research Lab - Built with Streamlit & Hugging Face Transformers")
