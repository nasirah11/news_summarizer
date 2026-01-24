!pip install transformers sentencepiece nltk rouge-score

from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from rouge_score import rouge_scorer

nltk.download('punkt')

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("Model loaded successfully!")

def generate_summary(text, max_len=80, min_len=30):
    input_text = "summarize: " + text

    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

article_text = """
The government has announced a new policy aimed at improving digital education in schools.
The initiative will provide funding for technology infrastructure and teacher training.
Officials believe the program will enhance learning outcomes and reduce the digital divide.
Several education experts welcomed the move, stating that digital skills are essential for future careers.
However, some critics argued that rural schools may still face challenges in accessing high-speed internet.
"""

summary = generate_summary(article_text)

print("ORIGINAL ARTICLE:\n", article_text)
print("\nGENERATED SUMMARY:\n", summary)

!unzip "BBC News Summary.zip"

import os

def load_bbc_samples(article_dir, summary_dir, limit=3):
    articles = []
    references = []

    categories = os.listdir(article_dir)
    count = 0

    for category in categories:
        art_path = os.path.join(article_dir, category)
        sum_path = os.path.join(summary_dir, category)

        files = os.listdir(art_path)

        for file in files:
            if count >= limit:
                break

            with open(os.path.join(art_path, file), "r", encoding="latin-1") as f:
                article = f.read().replace("\n", " ")

            with open(os.path.join(sum_path, file), "r", encoding="latin-1") as f:
                summary = f.read().replace("\n", " ")

            articles.append(article)
            references.append(summary)
            count += 1

    return articles, references

articles, references = load_bbc_samples(
    "BBC News Summary/News Articles",
    "BBC News Summary/Summaries",
    limit=3
)

print("Total loaded:", len(articles))
print("\nFirst article (short):\n", articles[0][:500])
print("\nReference summary:\n", references[0])

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# Summarize the first article
summary_result = summarizer(articles[0], max_length=100, min_length=30, do_sample=False)
print("Generated summary:\n", summary_result[0]['summary_text'])

print("Human summary:\n", references[0])

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(references[0], summary_result[0]['summary_text'])
print(scores)

from transformers import pipeline
from rouge_score import rouge_scorer

# Load summarization pipeline
summarizer = pipeline("summarization")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Loop through all articles
for i, article in enumerate(articles):
    print(f"\n--- Article {i+1} ---")
    
    # Generate summary
    summary_result = summarizer(article, max_length=100, min_length=30, do_sample=False)
    generated_summary = summary_result[0]['summary_text']
    
    # Print summaries
    print("Generated summary:\n", generated_summary)
    print("Human summary:\n", references[i])
    
    # Calculate ROUGE
    scores = scorer.score(references[i], generated_summary)
    print("ROUGE scores:\n", scores)

