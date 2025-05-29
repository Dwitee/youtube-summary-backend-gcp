import os
import requests
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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


def generate_mindmap_transformer(summary_text):

    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    print(f"[DEBUG] Prompt for transformer model with flan-t5-large  :\n{prompt}")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map="auto")
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    raw_result = summarizer(prompt, max_new_tokens=256, do_sample=True)
    print(f"[DEBUG] Raw transformer model output:\n{raw_result}")
    result = raw_result[0]["generated_text"]
    print(f"[DEBUG] Transformer model response:\n{result}")

    try:
        json_match = re.search(r'({[^{}]*"central"[^{}]*"branches"[^{}]*{.*?}[^{}]*})', result, re.DOTALL)
        if not json_match:
            raise ValueError("Mind map JSON not found in transformer model output.")
        json_str = json_match.group(1)
        print(f"[DEBUG] Extracted mind map JSON:\n{json_str}")
        return json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to parse transformer model output: {e}")