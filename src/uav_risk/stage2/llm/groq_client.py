from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import requests


@dataclass
class GroqClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 60

    @classmethod
    def from_env(cls) -> "GroqClient":
        api_key = os.environ.get("GROQ_API_KEY", "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").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")
        base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        model = os.environ.get("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905").strip()
        return cls(api_key=api_key, base_url=base_url, model=model)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1, top_p: float = 0.9) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
