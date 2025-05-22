
from transformers import pipeline

# Choose a lightweight model for faster CPU response, or change to bart-large-cnn for better summaries
summarizer = pipeline("summarization", model="t5-small")

def summarize_text(text):
    text = text.strip()
    if len(text.split()) > 400:
        text = " ".join(text.split()[:400])  # Truncate to 400 words
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']
