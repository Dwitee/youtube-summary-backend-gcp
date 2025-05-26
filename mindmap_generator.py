import os
import requests
import json

# Set your Hugging Face token as an environment variable: HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable not set.")
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

PROMPT_TEMPLATE = """
You are a helpful assistant. Convert the summary below into a mind map JSON with:
- A central topic
- 3 to 5 key branches
- 2 to 4 subpoints per branch

Format your response as:
{
  "central": "Main Topic",
  "branches": {
    "Branch 1": ["Point A", "Point B"],
    ...
  }
}

Summary:
\"\"\"
{summary}
\"\"\"
"""
def extract_json(text):
    """Extract the first valid JSON object from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return text  # Return raw if parsing fails
    return text  # Fallback to raw
def generate_mindmap_structure(summary_text):
    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    data = {"inputs": prompt}

    response = requests.post(MODEL_URL, headers=HEADERS, json=data)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            raw_output = result[0]["generated_text"]
            return extract_json(raw_output)
        return result
    else:
        raise Exception(f"Hugging Face API call failed: {response.status_code} — {response.text}")