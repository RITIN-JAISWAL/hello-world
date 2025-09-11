# src/personas/loader.py
from __future__ import annotations
import os, yaml
from typing import Dict, Any

def load_personas(path: str | None = None) -> Dict[str, Any]:
    """
    Load persona definitions from YAML.
    Returns dict: { persona_id: {display_name, style, ...}, ... }
    """
    path = path or os.path.join(os.path.dirname(__file__), "personas.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("personas", {})

def persona_system_prompt(style_text: str) -> str:
    """
    Build a system prompt that asks the LLM to rewrite answers in the persona style.
    """
    return (
        "You are a metering domain assistant. Rewrite the answer in the following style. "
        "Keep all facts intact; do not invent data. Style:\n" + (style_text or "").strip()
    )
