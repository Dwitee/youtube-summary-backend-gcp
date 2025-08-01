from transformers import pipeline
import re

CHAPTERIZE_PROMPT_TEMPLATE = """
Chapterize the content by dividing it into as many meaningful chapters as appropriate based on topic shifts, changes in speaker, or major events. For a 20-minute video, aim for at least 5–8 chapters if possible.
For each chapter, provide:
  1. A short, descriptive title.
  2. The timestamp where this chapter starts, in the format mm:ss or HH:MM:SS.
  3. A detailed summary covering the most important points, key takeaways, and any relevant facts or arguments.

Avoid speculation or made-up information. The output must be strictly a JSON array of objects, where each object contains:
  - "chapterTitle": a brief heading for the chapter.
  - "startTime": the start time of the chapter in the video.
  - "chapterSummary": a more elaborate summary of that chapter's content (3–5 sentences recommended).

Example output:
[
  {{
    "chapterTitle": "Chapter 1: Introduction to Machine Learning",
    "startTime": "00:00:55",
    "chapterSummary": "This chapter provides an overview of machine learning, outlining its goals, common algorithms, and real-world applications. It sets the foundation for understanding supervised and unsupervised learning approaches."
  }},
  {{
    "chapterTitle": "Chapter 2: Supervised vs Unsupervised Learning",
    "startTime": "00:05:20",
    "chapterSummary": "This chapter covers the difference between supervised and unsupervised learning, discussing typical algorithms like regression and clustering, and explaining when to use each approach."
  }}
]

Content:
\"\"\"
{content}
\"\"\"
"""

# Choose a lightweight model for faster CPU response, or change to bart-large-cnn for better summaries
summarizer = pipeline("summarization", model="t5-small")

def summarizer_gemini(text):
    text = text.strip()
    from google import genai
    import json
    print(f"[DEBUG] About to summarize content using Gemini summarizer...")

    genai_client = genai.Client(
        vertexai=True,
        project="secure-garden-460600-u4",
        location="us-east4"
    )

    prompt = CHAPTERIZE_PROMPT_TEMPLATE.format(content=text)

    print(f"[DEBUG] Prompt for Gemini summarizer:\n{prompt}")

    chat = genai_client.chats.create(model="gemini-2.0-flash-001")
    response = chat.send_message(prompt)
    result = response.text.strip()
    # Remove Markdown-style ```json or ``` wrappers if present
    result = re.sub(r"^```(?:json)?\n", "", result)
    result = re.sub(r"\n```$", "", result)
    print(f"[DEBUG] Gemini summarizer response:\n{result}")

    try:
        json_data = json.loads(result)
        return json.dumps(json_data, indent=2)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Failed to parse Gemini summary JSON: {e}")

def summarize_text(text, model_name="gemini"):
    # summarized_text = summarize_t5_small(text)
    return summarizer_gemini(text) # sending summarized text to gemini to save tokens

    
def summarize_t5_small(text):
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