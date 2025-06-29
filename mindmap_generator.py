import os
import json
import re
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Set your Hugging Face token as an environment variable: HF_TOKEN
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise EnvironmentError("HF_TOKEN environment variable not set.")

# PROMPT_TEMPLATE = """
# You are a helpful assistant. Convert the summary below into a mind map JSON with:
# - A central topic
# - 3 to 5 key branches
# - 2 to 4 subpoints per branch
# - Each branch and subpoint label should include a relevant emoji based on the topic (e.g., 📘 for Introduction, 💡 for Ideas, 🧠 for Neural Networks, 📊 for Data, etc.)
# Format your response as:
# {{
#   "central": "Main Topic",
#   "branches": {{
#     "Branch 1": ["Point A", "Point B"],
#     ...
#   }}
# }}

# Output ONLY the JSON. Do not include any explanations, prefixes, or formatting.

# Summary:
# \"\"\"
# {summary}
# \"\"\"
# """


PROMPT_TEMPLATE = """
You are a helpful assistant. Convert the summary below into a mind map JSON with:
- A central topic
- 3 to 5 key branches
- 2 to 4 subpoints per branch
- Each topic (central, branches, and points) must include:
  - "label": the title with a relevant emoji
  - "narration": a simple sentence that explains concept of this label with respect to the parent node and the whole summary

Format your response as:
{{
  "central": {{
    "label": "🧠 Main Topic",
    "narration": "This is the central idea explained simply."
  }},
  "branches": [
    {{
      "label": "📘 Branch Title",
      "narration": "In this sub-topic we cover...",
      "points": [
        {{
          "label": "💡 Subpoint A",
          "narration": "This talk about..."
        }},
        ...
      ]
    }},
    ...
  ]
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
    return 
    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map="auto")
    # summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    # raw_result = summarizer(prompt, max_new_tokens=256, do_sample=True)
    # print(f"[DEBUG] Raw transformer model output:\n{raw_result}")
    # result = raw_result[0].get("generated_text", "").strip()
    # print(f"[DEBUG] Transformer model response:\n{result}")

    # try:
    #     json_match = re.search(r'({[^{}]*"central"[^{}]*"branches"[^{}]*{.*?}[^{}]*})', result, re.DOTALL)
    #     if not json_match:
    #         raise ValueError("Mind map JSON not found in transformer model output.")
    #     json_str = json_match.group(1)
    #     print(f"[DEBUG] Extracted mind map JSON:\n{json_str}")
    #     return json.loads(json_str)
    # except Exception as e:
    #     raise RuntimeError(f"[ERROR] Failed to parse transformer model output: {e}")

def generate_mindmap_mistral(summary_text):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    print(f"[DEBUG] Prompt for Mistral model:\n{prompt}")
    return


def generate_mindmap_gemini(summary_text):
    from google import genai
    from google.genai import types

    genai_client = genai.Client(
        vertexai=True,
        project="secure-garden-460600-u4",
        location="us-east4",
    )

    prompt = PROMPT_TEMPLATE.format(summary=summary_text)
    print(f"[DEBUG] Prompt for Gemini GenAI SDK:\n{prompt}")

    chat = genai_client.chats.create(model="gemini-2.0-flash-001")
    response = chat.send_message(prompt)
    result = response.text.strip()
    print(f"[DEBUG] Gemini GenAI SDK response:\n{result}")

    # Updated regex to support deeply nested JSON with narration fields in central, branches, and points
    json_match = re.search(r'(\{.*"central".*"branches".*\}[\s\n]*)', result, re.DOTALL)
    if not json_match:
        raise ValueError("Mind map JSON not found in Gemini model output.")
    json_str = json_match.group(1)
    print(f"[DEBUG] Extracted mind map JSON:\n{json_str}")
    try:
        return json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to parse Gemini model output: {e}")