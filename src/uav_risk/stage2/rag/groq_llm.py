"""
Groq LLM Integration for UAV RAG System
========================================
دمج Groq API مع نظام RAG للحصول على استجابات سريعة وقوية
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from groq import AsyncGroq

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger("GroqLLM")

@dataclass
class GroqLLMConfig:
    """إعدادات Groq API"""
    api_key: str
    model: str = "llama-3.3-70b-versatile"  # نماذج متاحة: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
    temperature: float = 0.1  # منخفض للدقة القانونية
    max_tokens: int = 4096
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class GroqLLM:
    """
    واجهة Groq LLM للنظام
    """
    
    def __init__(self, config: GroqLLMConfig):
        self.config = config
        self.client = AsyncGroq(api_key=config.api_key)
        self.system_prompt = SYSTEM_PROMPT
    
    async def generate(self, prompt: str, include_system: bool = True) -> str:
        """توليد رد من النموذج"""
        
        messages = []
        if include_system:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return f"خطأ في الاتصال بـ Groq API: {e}"
    
    async def generate_with_context(self, query: str, context: str) -> str:
        """توليد رد مع سياق قانوني"""
        
        prompt = f"""Based on the following legal context, answer the user's question.

LEGAL CONTEXT:
{context}

USER QUESTION: {query}

Remember to:
1. Cite specific sections
2. Compare FAA and EASA if both appear
3. Be precise and practical

ANSWER:"""
        
        return await self.generate(prompt, include_system=True)
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """تصنيف نوع السؤال"""
        
        from .prompts import QUERY_CLASSIFIER_PROMPT
        
        prompt = QUERY_CLASSIFIER_PROMPT.format(query=query)
        response = await self.generate(prompt, include_system=False)
        
        # تحليل الرد
        category = "OTHER"
        confidence = 0.5
        
        for line in response.split("\n"):
            if line.startswith("Category:"):
                category = line.replace("Category:", "").strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.replace("Confidence:", "").strip())
                except:
                    pass
        
        return {"category": category, "confidence": confidence}
    
    async def generate_hypothetical_answer(self, query: str) -> str:
        """توليد إجابة افتراضية لـ HyDE"""
        
        from .prompts import HYDE_PROMPT
        
        prompt = HYDE_PROMPT.format(query=query)
        return await self.generate(prompt, include_system=False)