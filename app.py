import streamlit as st
from transformers import pipeline
import os
from pathlib import Path

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

st.title("🛡️ SpamGuard AI")
st.caption("Developed by Muhammad Hamza • Qassim University")

# --- 2. DEBUGGING / DIAGNOSTIC (Delete this section once it works) ---
with st.expander("🔍 System Diagnostic (Check this if it fails)"):
    st.write("Current Directory Files:", os.listdir("."))
    if os.path.exists("spam_model"):
        st.write("✅ 'spam_model' folder found.")
        st.write("Files inside 'spam_model':", os.listdir("spam_model"))
    else:
        st.error("❌ 'spam_model' folder NOT found at root level.")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_spam_model():
    model_dir = "spam_model"
    
    # Check for the actual weight file before loading
    weight_path = os.path.join(model_dir, "model.safetensors")
    
    if not os.path.exists(weight_path):
        return None, "Weights file (model.safetensors) missing."

    try:
        # Load the model and tokenizer from the local folder
        pipe = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            device=-1  # Forces CPU
        )
        return pipe, "Success"
    except Exception as e:
        return None, str(e)

# Initialize
classifier, message = load_spam_model()

# --- 4. MAIN INTERFACE ---
user_input = st.text_area("Paste Message Here:", height=150, placeholder="Enter text...")

if st.button("Analyze Now", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter text first.")
    elif classifier is None:
        st.error(f"Model Error: {message}")
        st.info("Tip: Ensure your model files are in a folder named 'spam_model' on GitHub.")
    else:
        with st.spinner("Classifying..."):
            result = classifier(user_input[:512])[0]
            label = result['label']
            score = result['score']
            
            # Final Output
            st.divider()
            # Update your code to this:
if label == "LABEL_0":
    st.error("### 🚨 SPAM DETECTED")
else:
    st.success("### ✅ SAFE MESSAGE")
            
            with st.expander("Technical Data"):
                st.json(result)
