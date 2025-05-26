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
        print(f"[DEBUG] Response Text: {response.text}")
    except Exception as e:
        raise Exception(f"[ERROR] Exception while calling Zephyr: {e}")

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        raise Exception(f"Hugging Face API call failed: {response.status_code} — {response.text}")