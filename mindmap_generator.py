import os
import requests
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Set your Hugging Face token as an environment variable: HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable not set.")

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
    result = raw_result[0].get("generated_text", "").strip()
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

def generate_mindmap_mistral(summary_text):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    print(f"[DEBUG] Prompt for Mistral model:\n{prompt}")

    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 quantization_config=quantization_config, token=HF_TOKEN)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    output = generator(prompt, max_new_tokens=512, do_sample=False, return_full_text=False)
    print(f"[DEBUG] Raw Mistral model output:\n{output}")
    result = output[0].get("generated_text", "").strip()
    print(f"[DEBUG] Mistral model response:\n{result}")

    try:
        json_match = re.search(r'({[^{}]*"central"[^{}]*"branches"[^{}]*{.*?}[^{}]*})', result, re.DOTALL)
        if not json_match:
            raise ValueError("Mind map JSON not found in Mistral model output.")
        json_str = json_match.group(1)
        print(f"[DEBUG] Extracted mind map JSON:\n{json_str}")
        return json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to parse Mistral model output: {e}")