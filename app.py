import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
st.write("You can paste a news article OR upload a text file, then click **Summarize**.")

# Option selection
option = st.radio("Choose input method:", ["Paste Text", "Upload Text File"])

article_text = ""

# Paste text option
if option == "Paste Text":
    article_text = st.text_area("Enter News Article Text Here:", height=300)

# Upload file option
elif option == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a .txt file containing a news article", type=["txt"])

    if uploaded_file is not None:
        article_text = uploaded_file.read().decode("utf-8")
        st.subheader("📄 Uploaded Article Preview")
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
