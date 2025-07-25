from transformers import pipeline
import re

CHAPTERIZE_PROMPT_TEMPLATE = """
Chapterize the content by dividing it into at least two meaningful chapters based on topic shifts or major events. 
For each chapter, provide:
1. A short, descriptive title.
2. The timestamp where this chapter starts, in the format mm:ss or HH:MM:SS.
3. A detailed summary covering the most important points, key takeaways, and any relevant facts or arguments.

Avoid speculation or made-up information. The format should be strictly a JSON array of objects, where each object contains:
- "chapterTitle": a brief heading for the chapter.
- "startTime": the start time of the chapter in the video.
- "chapterSummary": a more elaborate summary of that chapter's content (3–5 sentences recommended).

Example format:
[
  {
    "chapterTitle": "Chapter 1: Introduction to Machine Learning",
    "startTime": "00:00:55",
    "chapterSummary": "This chapter provides an overview of machine learning, outlining its goals, common algorithms, and real-world applications. It sets the foundation for understanding supervised and unsupervised learning approaches."
  }
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
    print(f"[DEBUG] summarizer_gemini input text (first 500 chars):\n{text[:500]!r}", flush=True)
    from google import genai
    import json
    print(f"[DEBUG] About to summarize content using Gemini summarizer...", flush=True)
    print("[DEBUG] Initializing Gemini GenAI client...", flush=True)

    genai_client = genai.Client(
        vertexai=True,
        project="secure-garden-460600-u4",
        location="us-east4"
    )
    print("[DEBUG] GenAI client initialized", flush=True)

    prompt = CHAPTERIZE_PROMPT_TEMPLATE.format(content=text)
    print(f"[DEBUG] Prompt for Gemini summarizer:\n{prompt}", flush=True)
    print("[DEBUG] Creating chat object...", flush=True)

    chat = genai_client.chats.create(model="gemini-2.0-flash-001")
    print(f"[DEBUG] Chat object created: {chat}", flush=True)
    print("[DEBUG] Sending prompt to Gemini...", flush=True)

    response = chat.send_message(prompt)
    print(f"[DEBUG] Received raw response object: {response}", flush=True)
    print(f"[DEBUG] Received raw response text (first 200 chars): {response.text[:200]!r}", flush=True)

    result = response.text.strip()
    # Remove Markdown-style ```json or ``` wrappers if present
    result = re.sub(r"^```(?:json)?\n", "", result)
    result = re.sub(r"\n```$", "", result)
    # Debug: inspect cleaned result
    print(f"[DEBUG] Gemini summarizer response (stripped):")
    print(f"  Type: {type(result)}")
    print(f"  Length: {len(result)} characters")
    print(f"  Preview: {result[:200]!r}")  # show repr of first 200 chars
    print(f"[DEBUG] End of preview")
    import sys
    sys.stdout.flush()

    try:
        json_data = json.loads(result)
        return json.dumps(json_data, indent=2)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse Gemini summary JSON: {e}")
        print(f"[ERROR] Full result causing parse failure:\n{result!r}")
        raise RuntimeError(f"[ERROR] Failed to parse Gemini summary JSON: {e}")

def summarize_text(text, model_name="gemini"):
    summarized_text = summarize_t5_small(text)
    print(f"[DEBUG] T5 summarized text:\n{summarized_text}")
    return summarizer_gemini(summarized_text) # sending summarized text to gemini to save tokens

    
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