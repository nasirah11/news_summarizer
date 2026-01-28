import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="News Article Summarizer",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("📰 News Article Summarizer")
st.markdown(
    """
    A simple **Natural Language Processing (NLP) application** that automatically  
    generates concise summaries from news articles using a transformer model.
    """
)

st.divider()

# -----------------------------
# Sidebar (Input Options)
# -----------------------------
st.sidebar.header("⚙️ Input Options")
input_method = st.sidebar.radio(
    "Choose how you want to provide the news article:",
    ["Paste Text", "Upload PDF"]
)

st.sidebar.markdown(
    """
    **Supported formats:**
    - Paste plain text  
    - Upload a PDF news article  

    
)

# -----------------------------
# Main Input Area
# -----------------------------
article_text = ""

if input_method == "Paste Text":
    st.subheader("📄 Paste News Article")
    article_text = st.text_area(
        "Paste the full news article text below:",
        height=280,
        placeholder="Paste your news article here..."
    )

elif input_method == "Upload PDF":
    st.subheader("📑 Upload PDF News Article")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            article_text = " ".join([p for p in pages if p])

        st.markdown("**Extracted Text Preview:**")
        st.info(article_text[:1000] + "...")

# -----------------------------
# Summarize Button
# -----------------------------
st.divider()

if st.button("✨ Generate Summary"):
    if article_text.strip() == "":
        st.warning("Please provide a news article before summarizing.")
    else:
        with st.spinner("Generating summary, please wait... ⏳"):
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

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        st.subheader("✂️ Generated Summary")
        st.success(summary)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Developed as an academic NLP project using Streamlit and BART transformer model."
)
