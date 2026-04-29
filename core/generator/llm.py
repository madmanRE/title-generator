import os
import dotenv
from openai import OpenAI
import streamlit as st

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

def dict_to_text(d):
    lines = []
    lines.append(f"head_phrase: {d.get('head_phrase', '')}")
    mandatory = d.get("mandatory_phrases", [])
    lines.append("mandatory_phrases:")
    for m in mandatory:
        lines.append(f"- {m}")
    optional = d.get("optional_phrases", [])
    lines.append("optional_phrases:")
    for o in optional:
        lines.append(f"- {o}")
    synonyms = d.get("synonyms", {})
    lines.append("synonyms:")
    for k, v in synonyms.items():
        if v:
            lines.append(f"- {k}: {v}")
        else:
            lines.append(f"- {k}: []")
    return "\n".join(lines)

def generate_title(content, prompt, model, strategy="GPT"):
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": content.to_string() if strategy == "GPT" else dict_to_text(content),
        },
    ]
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content
