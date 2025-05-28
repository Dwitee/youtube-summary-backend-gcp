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

def generate_mindmap_from_gguf(summary_text):
    from llama_cpp import Llama

    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    print(f"[DEBUG] Prompt for GGUF model:\n{prompt}")

    model_path = "./models/zephyr.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8)

    response = llm(prompt, max_tokens=1024, stop=["</s>", "```"], echo=False)
    result = response["choices"][0]["text"].strip()
    print(f"[DEBUG] GGUF model response:\n{result}")

    try:
        json_match = re.search(r'({[^{}]*"central"[^{}]*"branches"[^{}]*{.*?}[^{}]*})', result, re.DOTALL)
        if not json_match:
            raise ValueError("Mind map JSON not found in GGUF model output.")
        json_str = json_match.group(1)
        print(f"[DEBUG] Extracted mind map JSON:\n{json_str}")
        return json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to parse GGUF model output: {e}")