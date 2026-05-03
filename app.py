import streamlit as st
from transformers import pipeline
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

# --- CUSTOM STYLING (Optional) ---
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 1.2rem !important; }
    .main-title { text-align: center; color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown("<h1 class='main-title'>🛡️ SpamGuard AI</h1>", unsafe_allow_html=True)
st.write("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = Path("spam_model")
    if not model_path.exists():
        return None
    try:
        # device=-1 ensures it runs on CPU for Streamlit Cloud
        return pipeline("text-classification", model=str(model_path), tokenizer=str(model_path), device=-1)
    except:
        return None

classifier = load_model()

# --- INPUT SECTION ---
user_input = st.text_area("Paste your message below:", height=150, placeholder="Enter email or SMS text...")

# --- BUTTON ---
analyze_btn = st.button("🔍 Scan for Threats", type="primary", use_container_width=True)

# --- OUTPUT SECTION ---
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a message first.")
    elif classifier is None:
        st.error("❌ Error: Model files not found in 'spam_model' folder.")
    else:
        with st.spinner("AI is analyzing..."):
            result = classifier(user_input[:512])[0]
            label = result['label']
            score = result['score']
            
            # --- RESULTS DISPLAY ---
            st.write("### 📊 Analysis Results")
            
            # Determine if Spam (Adjust 'LABEL_1' if your model uses 'LABEL_0' for spam)
            if label == "LABEL_1":
                st.error(f"**Result:** This message is likely **SPAM**")
                st.progress(score)
                st.write(f"**AI Confidence:** {score:.1%}")
            else:
                st.success(f"**Result:** This message appears to be **SAFE**")
                st.progress(score)
                st.write(f"**AI Confidence:** {score:.1%}")
            
            # Simple metadata view
            with st.expander("Technical Data"):
                st.json(result)

# --- FOOTER ---
st.write("---")
st.caption("Developed by Muhammad Hamza • Qassim University • MS in AI")
