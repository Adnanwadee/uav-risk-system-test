"""
Module: src/uav_risk/stage2/rag/groq_llm.py
Author: Elite Technical Partner
Description: Asynchronous API client wrapper for Groq LLM inference, optimized for aviation compliance.
"""

import asyncio
from typing import Any, Dict, List, Optional
import structlog
from groq import AsyncGroq

# الاستيراد المطلق للعقود والموجهات الموحدة بناءً على قواعد الاتساق
from uav_risk.stage2.rag.config import GroqLLMConfig
from uav_risk.stage2.rag.prompts import SYSTEM_PROMPT, QUERY_CLASSIFIER_PROMPT, HYDE_PROMPT

logger = structlog.get_logger()


class GroqLLM:
    """Production-grade asynchronous interface for Groq engine client operations."""

    def __init__(self, config: GroqLLMConfig):
        self.config = config
        self.client = AsyncGroq(api_key=config.api_key)
        self.system_prompt = SYSTEM_PROMPT
        logger.info("groq_llm_client_initialized", model=config.model, temperature=config.temperature)

    async def generate(self, prompt: str, include_system: bool = True) -> str:
        """Generates a text completion from Groq API with defensive timeout handling."""
        logger.debug("groq_inference_request_sent", prompt_len=len(prompt), include_system=include_system)
        
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
            output_text = response.choices[0].message.content
            logger.debug("groq_inference_response_received", output_len=len(output_text))
            return output_text
        except Exception as exc:
            logger.error("groq_api_call_failed_fallback_activated", error=str(exc))
            # Raise so upstream callers (ReportWriter, agents) can detect failure
            # and activate fallback behavior instead of silently receiving text.
            raise RuntimeError(f"Error connecting to Groq API client pipeline: {str(exc)}")

    async def generate_with_context(self, query: str, context: str) -> str:
        """Assembles a high-density grounded context block to answer the compliance query."""
        structured_prompt = (
            f"Based on the following verified legal context, answer the user's aviation compliance question.\n\n"
            f"LEGAL CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Remember to strictly apply constraints, compare FAA/EASA if relevant, and remain entirely factual.\n"
            f"ANSWER:"
        )
        return await self.generate(structured_prompt, include_system=True)

    async def classify_query(self, query: str) -> dict[str, Any]:
        """Classifies the incoming operational request into discrete aviation categories."""
        formatted_prompt = QUERY_CLASSIFIER_PROMPT.format(query=query)
        response_text = await self.generate(formatted_prompt, include_system=False)
        
        # التنقيب البرمجي عن التصنيف ونسبة الثقة المستخرجة من الرد الاستدلالي
        category = "OTHER"
        confidence = 0.5
        
        for line in response_text.split("\n"):
            cleaned_line = line.strip()
            if cleaned_line.startswith("Category:"):
                category = cleaned_line.replace("Category:", "").strip()
            elif cleaned_line.startswith("Confidence:"):
                try:
                    confidence = float(cleaned_line.replace("Confidence:", "").strip())
                except (ValueError, TypeError):
                    pass
                    
        logger.info("query_classification_concluded", category=category, confidence=confidence)
        return {"category": category, "confidence": confidence}

    async def generate_hypothetical_answer(self, query: str) -> str:
        """Generates an idealized regulatory passage to feed the HyDE search module."""
        formatted_prompt = HYDE_PROMPT.format(query=query)
        return await self.generate(formatted_prompt, include_system=False)


# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# Core LLM gateway driver interacting with external API clusters.
# Dependencies: src/uav_risk/stage2/rag/config.py -> GroqLLMConfig
#               src/uav_risk/stage2/rag/prompts.py -> Prompts Registries
# Dependent Files: src/uav_risk/stage2/rag/enhanced_retriever.py, rag_core.py
# =====================================================================