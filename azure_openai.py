# src/llm/azure_openai.py
from __future__ import annotations
import os
from openai import AzureOpenAI

_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

_client = AzureOpenAI(api_key=_API_KEY, api_version=_API_VERSION, azure_endpoint=_ENDPOINT)

def rewrite_with_persona(raw_text: str, persona_sys_prompt: str, max_tokens: int = 450) -> str:
    """
    Persona-only rewriting. If AOAI env isn't present, return raw_text unchanged.
    """
    if not (_API_KEY and _ENDPOINT and _DEPLOYMENT):
        return raw_text
    resp = _client.chat.completions.create(
        model=_DEPLOYMENT,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": persona_sys_prompt},
            {"role": "user", "content": raw_text},
        ],
    )
    return (resp.choices[0].message.content or raw_text).strip()
