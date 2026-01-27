import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer (only once)
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# App title
st.title("📰 News Article Summarizer")
st.write("Paste a news article below and click **Summarize** to generate a short summary.")

# Input text
article_text = st.text_area("Enter News Article Text Here:", height=300)

# Button
if st.button("Summarize"):
    if article_text.strip() == "":
        st.warning("Please paste a news article first.")
    else:
        with st.spinner("Generating summary... Please wait ⏳"):

            inputs = tokenizer(
                article_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=60,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Output
        st.subheader("✂️ Generated Summary")
        st.success(summary)
