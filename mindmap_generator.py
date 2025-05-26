import os
import requests
import json
import re

# Set your Hugging Face token as an environment variable: HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable not set.")
MODEL_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

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
{{
  "central": "Main Topic",
  "branches": {{
    "Branch 1": ["Point A", "Point B"],
    ...
  }}
}}

Output ONLY the JSON. Do not include any explanations, prefixes, or formatting.

Summary:
\"\"\"
{summary}
\"\"\"
"""

def generate_mindmap_structure(summary_text):
    print(f"XXXXXXXXXXXXXXXXXX")
    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    data = {"inputs": prompt}

    print("[DEBUG] Prompt being sent to Zephyr:")
    print(prompt)

    try:
        response = requests.post(MODEL_URL, headers=HEADERS, json=data)
        print(f"[DEBUG] Raw response object: {response}")
        print(f"[DEBUG] Status Code: {response.status_code}")
    except Exception as e:
        raise Exception(f"[ERROR] Exception while calling Zephyr: {e}")

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            text = result[0]["generated_text"]
            print(f"[DEBUG] Full Zephyr generated text:\n{text}")

            # Try to extract JSON from common response format
            json_match = re.search(r'(?:Convert to JSON:|JSON)?\s*({.*})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                print(f"[DEBUG] Extracted JSON string:\n{json_str}")
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to decode JSON: {e}")
            else:
                raise ValueError("Mind map JSON not found in Zephyr output.")
        else:
            raise ValueError("Unexpected response format from Hugging Face.")
    else:
        raise Exception(f"Hugging Face API call failed: {response.status_code} — {response.text}")