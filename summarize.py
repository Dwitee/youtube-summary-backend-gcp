from transformers import pipeline

# Choose a lightweight model for faster CPU response, or change to bart-large-cnn for better summaries
summarizer = pipeline("summarization", model="t5-small")

def summarize_text(text):
    text = text.strip()
    words = text.split()
    chunk_size = 400
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    summaries = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        result = summarizer(chunk_text, max_length=130, min_length=30, do_sample=False)
        summaries.append(result[0]['summary_text'])

    final_summary = " ".join(summaries)
    return final_summary
