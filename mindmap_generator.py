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

def generate_mindmap_zephyr_locally(summary_text, model, tokenizer, device="cuda"):
    from transformers import pipeline

    print("[DEBUG] Generating mind map using local Zephyr model...")

    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

    try:
        result = pipe(prompt, max_new_tokens=1024, do_sample=True)[0]["generated_text"]
        print(f"[DEBUG] Full local Zephyr generated text:\n{result}")

        json_match = re.search(r'Your response:\s*({[^{}]*"central"[^{}]*"branches"[^{}]*{.*?}[^{}]*})', result, re.DOTALL)
        if not json_match:
            json_blocks = re.findall(r'({[^{}]*"central"[^{}]*"branches"[^{}]*{.*?}[^{}]*})', result, re.DOTALL)
            json_blocks = [block for block in json_blocks if '"Main Topic"' not in block and '"Branch 1"' not in block]
            if json_blocks:
                json_str = json_blocks[-1]
            else:
                raise ValueError("Mind map JSON not found in Zephyr local output.")
        else:
            json_str = json_match.group(1)

        print(f"[DEBUG] Extracted local JSON string:\n{json_str}")
        return json.loads(json_str)

    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to generate mind map locally: {e}")