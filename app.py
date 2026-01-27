import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

st.title("📰 News Article Summarizer")
st.write("Paste a news article below and click **Summarize** to generate a short summary.")

article_text = st.text_area("Enter News Article Text Here:", height=300)

if st.button("Summarize"):
    if article_text.strip() == "":
        st.warning("Please paste a news article first.")
    else:
        with st.spinner("Generating summary... Please wait ⏳"):
            summary = summarizer(
                article_text[:1024],
                max_length=150,
                min_length=60,
                do_sample=False
            )[0]['summary_text']

        st.subheader("✂️ Generated Summary")
        st.success(summary)
