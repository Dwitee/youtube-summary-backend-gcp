from transformers import pipeline

CHAPTERIZE_PROMPT_TEMPLATE = """
Chapterize the content by grouping the content into chapters and providing a summary for each chapter.
Please only capture key events and highlights. If you are not sure about any info, please do not make it up. 
Return the result strictly as a JSON array of objects, without any commentary or extra text. Each object must contain two keys:
- "chapterTitle": a short title or heading for the chapter.
- "chapterSummary": a concise summary of that chapter.

Example format:
[
  {{
    "chapterTitle": "Chapter 1: Introduction",
    "chapterSummary": "This chapter introduces the main topic and outlines what will be covered."
  }}
]

"""
{text}
"""
""".strip()

# Choose a lightweight model for faster CPU response, or change to bart-large-cnn for better summaries
summarizer = pipeline("summarization", model="t5-small")

def summarizer_gemini(text):
    from google import genai
    import json
    print(f"[DEBUG] About to summarize content using Gemini summarizer...")

    genai_client = genai.Client(
        vertexai=True,
        project="secure-garden-460600-u4",
        location="us-east4"
    )

    prompt = CHAPTERIZE_PROMPT_TEMPLATE.format(text=text)

    print(f"[DEBUG] Prompt for Gemini summarizer:\n{prompt}")

    chat = genai_client.chats.create(model="gemini-2.0-flash-001")
    response = chat.send_message(prompt)
    result = response.text.strip()
    print(f"[DEBUG] Gemini summarizer response:\n{result}")

    try:
        json_data = json.loads(result)
        return json.dumps(json_data, indent=2)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Failed to parse Gemini summary JSON: {e}")

def summarize_text(text, model_name="t5-small"):
    text = text.strip()
    
    if model_name.lower() == "gemini":
        return summarizer_gemini(text)

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
