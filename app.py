import streamlit as st
from transformers import pipeline
import torch
from pathlib import Path

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="centered")

st.title("🛡️ SpamGuard AI")
st.subheader("Intelligent Spam & Phishing Detector")

# Load Model
@st.cache_resource
def load_model():
    model_path = Path("spam_model")
    if not model_path.exists():
        st.error("❌ Model folder 'spam_model' not found!")
        return None
    
    try:
        pipe = pipeline(
            "text-classification",
            model=str(model_path),
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("✅ Model Loaded Successfully")
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# User Input
user_input = st.text_area(
    "Paste Email or SMS Message Here",
    height=180,
    placeholder="Enter message to analyze..."
)

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("AI is analyzing the message..."):
            try:
                result = model(user_input[:512])[0]
                
                is_spam = result['label'].lower() in ['spam', '1', 'label_1']
                confidence = result['score']
                
                st.divider()
                
                if is_spam:
                    st.error("### 🚨 SPAM / PHISHING DETECTED", icon="⚠️")
                    st.write(f"**Confidence:** `{confidence:.1%}`")
                    st.warning("**Recommended Action:** Do not reply or click any links.")
                else:
                    st.success("### ✅ This message appears safe", icon="✅")
                    st.write(f"**Confidence:** `{confidence:.1%}`")
                
                with st.expander("See Raw AI Output"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.warning("Please enter a message.")

st.markdown("---")
st.caption("SpamGuard AI • Built with DistilBERT")
