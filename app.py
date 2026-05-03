import streamlit as st
from transformers import pipeline
import torch
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="centered")

st.title("🛡️ SpamGuard AI")
st.subheader("Intelligent Spam & Phishing Detector")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = Path("spam_model")
    
    if not model_path.exists():
        st.error("❌ Model folder 'spam_model' not found! Please check your repository structure.")
        return None
    
    try:
        # Change this part in your load_model function:
        pipe = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            local_files_only=True  # This forces it to use YOUR files or fail
)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- APP LOGIC ---
if model is None:
    st.warning("⚠️ Application is waiting for the model to load...")
    st.stop()

# User Input
user_input = st.text_area(
    "Paste Email or SMS Message Here",
    height=200,
    placeholder="Enter message to analyze (e.g., urgent prize notifications or work emails)..."
)

if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Analyzing with DistilBERT..."):
            try:
                # Get prediction (truncating to 512 tokens for model safety)
                result = model(user_input[:512])[0]
                
                label = result['label'].upper()
                confidence = result['score']
                
                # REVISED LOGIC: 
                # Check for common naming conventions: 'LABEL_1', 'SPAM', or '1'
                # Note: If your model treats LABEL_0 as spam, change this line.
                is_spam = label in ['LABEL_0', '0']
                
                st.divider()
                
                if is_spam:
                    st.error(f"### 🚨 SPAM / PHISHING DETECTED", icon="⚠️")
                    st.write(f"**AI Confidence:** `{confidence:.1%}`")
                    st.warning("**Recommended Action:** This message looks suspicious. Do not click links or share personal info.")
                else:
                    st.success(f"### ✅ This message appears safe", icon="✅")
                    st.write(f"**AI Confidence:** `{confidence:.1%}`")

                # Technical Debugging Section
                with st.expander("Technical Details (Research Data)"):
                    st.write(f"**Detected Label:** `{label}`")
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Analysis Error: {e}")
    else:
        st.warning("Please enter some text first.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Muhammad Hamza • MS in AI • Qassim University")
