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
    
    # Check if directory exists
    if not model_path.exists():
        st.error(f"❌ folder '{model_path}' not found at root.")
        return None
    
    # Check if weight file exists inside folder
    weight_file = model_path / "model.safetensors"
    if not weight_file.exists():
        st.error("❌ 'model.safetensors' missing from the folder! Check Git LFS.")
        return None

    try:
        # Load pipeline
        pipe = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=-1 # Forces CPU for stability on Streamlit Free Tier
        )
        return pipe
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# --- USER INTERFACE ---
user_input = st.text_area(
    "Paste Email or SMS Message Here",
    height=200,
    placeholder="Enter message to analyze..."
)

# Research Toggle: In case your labels are flipped (LABEL_0 vs LABEL_1)
is_label_1_spam = st.sidebar.toggle("Is LABEL_1 Spam?", value=True)

if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                result = model(user_input[:512])[0]
                label = result['label'].upper()
                score = result['score']
                
                # Dynamic Logic based on your sidebar selection
                if is_label_1_spam:
                    is_spam = "1" in label or "SPAM" in label
                else:
                    is_spam = "0" in label or "HAM" not in label # If 0 is spam

                st.divider()
                
                if is_spam:
                    st.error("### 🚨 SPAM DETECTED", icon="⚠️")
                else:
                    st.success("### ✅ MESSAGE SAFE", icon="✅")
                
                st.write(f"**Confidence:** `{score:.1%}`")
                
                with st.expander("Technical Research Data"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        st.warning("Please enter text.")

st.markdown("---")
st.caption("Muhammad Hamza • Qassim University • MS AI")
