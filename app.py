import streamlit as st
from transformers import pipeline
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

# Custom UI for a professional MS-AI project look
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 1.1rem !important; }
    .reportview-container { background: #f0f2f6; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SpamGuard AI")
st.markdown("### Intelligent Spam & Phishing Detection System")
st.caption("Researcher: Muhammad Hamza • Qassim University • MS in AI")
st.divider()

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = "spam_model"
    if not os.path.exists(model_path):
        return None
    try:
        # Load pipeline (Ensure model.safetensors is in the 'spam_model' folder)
        return pipeline("text-classification", model=model_path, tokenizer=model_path, device=-1)
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

classifier = load_model()

# --- 3. USER INPUT ---
user_input = st.text_area("Paste the email or message content here:", height=200, placeholder="e.g., 'Dear user, your account has been locked...'")

# --- 4. PROCESSING LOGIC ---
if st.button("🔍 Run AI Analysis", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please provide a message to analyze.")
    elif classifier is None:
        st.error("Model files not found. Check if 'spam_model' folder exists at the root.")
    else:
        with st.spinner("Analyzing linguistic patterns..."):
            # Get prediction
            result = classifier(user_input[:512])[0]
            label = result['label']
            score = result['score']
            
            st.subheader("Analysis Results")
            
            # --- THE "ALWAYS SAFE" FIX ---
            # Most models use LABEL_1 for Spam and LABEL_0 for Ham. 
            # If your testing shows they are flipped, change the "1" to "0" below.
            is_spam = ("1" in label or "SPAM" in label.upper())

            if is_spam:
                st.error(f"### 🚨 Result: HIGH RISK (SPAM)")
                st.metric("Spam Probability", f"{score:.1%}")
                st.write("⚠️ **Warning:** This message contains patterns typical of phishing or unsolicited spam.")
            else:
                st.success(f"### ✅ Result: LOW RISK (SAFE)")
                st.metric("Safety Confidence", f"{score:.1%}")
                st.write("✔️ **Info:** No malicious intent or spam patterns were detected.")

            # --- 5. TECHNICAL DATA (For Debugging) ---
            with st.expander("Technical Metadata (Research View)"):
                st.write("This section shows the raw output from your fine-tuned model.")
                st.json(result)
                st.info(f"Current detection logic: Spam = {label} (Targeting: LABEL_1)")

st.divider()
st.info("💡 **Researcher Tip:** If an obvious scam shows as 'SAFE', check the Technical Metadata. If it says 'LABEL_0', edit line 51 of this code to look for '0' instead of '1'.")
