import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber

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
st.write("Paste a news article OR upload a PDF file, then click **Summarize**.")

# Option selection
option = st.radio("Choose input method:", ["Paste Text", "Upload PDF File"])

article_text = ""

# Paste text option
if option == "Paste Text":
    article_text = st.text_area("Enter News Article Text Here:", height=300)

# Upload PDF option
elif option == "Upload PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file containing a news article", type=["pdf"])

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            article_text = " ".join([p for p in pages if p is not None])

        st.subheader("📄 Extracted Article Preview")
        st.write(article_text[:1000] + "...")

# Summarize button
if st.button("Summarize"):
    if article_text.strip() == "":
        st.warning("Please provide a news article first.")
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
