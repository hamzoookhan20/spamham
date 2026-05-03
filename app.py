import streamlit as st
from transformers import pipeline
import os

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

# Professional Styling
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 1.1rem !important; border-radius: 10px; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SpamGuard AI")
st.caption("Researcher: Muhammad Hamza • MS in AI Project")

# Model Loading Logic
@st.cache_resource
def load_spam_model():
    model_path = "spam_model"
    if not os.path.exists(model_path):
        return None, "Folder 'spam_model' missing."
    try:
        # device=-1 forces CPU (best for Streamlit Cloud)
        return pipeline("text-classification", model=model_path, tokenizer=model_path, device=-1), "Success"
    except Exception as e:
        return None, str(e)

classifier, status = load_spam_model()

user_input = st.text_area("Enter Message Content:", height=150)

if st.button("Analyze Now", type="primary"):
    if classifier:
        result = classifier(user_input[:512])[0]
        label = result['label']
        score = result['score']
        
        # TARGETING LABEL_0 AS SPAM
        if label == "LABEL_0":
            st.error(f"### 🚨 SPAM DETECTED ({score:.1%})")
        else:
            st.success(f"### ✅ SAFE MESSAGE ({score:.1%})")
            
        with st.expander("Technical View"):
            st.json(result)
    else:
        st.error(f"Model Error: {status}")
