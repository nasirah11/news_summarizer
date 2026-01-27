# BBC News Article Summarization using Transformer Model

This project implements an automatic news article summarization system using a pretrained transformer-based model (BART). The system generates abstractive summaries from news articles and evaluates the results using ROUGE metrics.

## Dataset
The BBC News Summary dataset is used in this project.  
Download from: https://www.kaggle.com/datasets/pariza/bbc-news-summary  

After downloading, extract the folder and place it in the project directory as:

BBC News Summary/
 ├── News Articles/
 └── Summaries/

## Requirements

Install dependencies using:

pip install -r requirements.txt

## How to Run

1. Ensure the dataset folder is placed correctly.
2. Run the program:

python main.py

## Output

The program will:
- Load and clean the news articles
- Generate summaries using BART
- Evaluate results using ROUGE-1, ROUGE-2, and ROUGE-L
- Display the evaluation table

## Author
Ra
Final Year Project – Automatic Text Summarization
