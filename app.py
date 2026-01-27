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

import re

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation & special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

clean_articles = [clean_text(a) for a in articles]
clean_references = [clean_text(s) for s in references]

print("BEFORE CLEANING:\n", articles[0][:200])
print("\nAFTER CLEANING:\n", clean_articles[0][:200])

!pip install transformers

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

generated_summaries = []

for i, article in enumerate(clean_articles):
    summary = summarizer(
        article[:1024],      # limit length (important!)
        max_length=150,
        min_length=60,
        do_sample=False
    )[0]['summary_text']

    generated_summaries.append(summary)

    print(f"\nARTICLE {i+1}")
    print("GENERATED SUMMARY:\n", summary)
    print("\nREFERENCE SUMMARY:\n", clean_references[i])

!pip install rouge-score

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for i in range(len(generated_summaries)):
    scores = scorer.score(clean_references[i], generated_summaries[i])

    print(f"\nARTICLE {i+1} ROUGE SCORES")
    print("ROUGE-1 F1:", scores['rouge1'].fmeasure)
    print("ROUGE-2 F1:", scores['rouge2'].fmeasure)
    print("ROUGE-L F1:", scores['rougeL'].fmeasure)

for i in range(len(generated_summaries)):
    print("\n==============================")
    print(f"ARTICLE {i+1}")
    print("GENERATED SUMMARY:\n", generated_summaries[i])
    print("\nREFERENCE SUMMARY:\n", clean_references[i])

results = []

for i in range(len(generated_summaries)):
    scores = scorer.score(clean_references[i], generated_summaries[i])

    results.append({
        "Article": i+1,
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    })

results

import pandas as pd

df = pd.DataFrame(results)
df
