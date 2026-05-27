"""
Module: src/uav_risk/stage2/rag/groq_llm.py
Author: Elite Technical Partner + V3.1 Production Fix
Description: Asynchronous API client wrapper for Groq LLM inference, 
             optimized for aviation compliance. Compatible with config_v3.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from groq import AsyncGroq

from .config_v3 import GroqLLMConfig
from .prompts_v3 import build_hyde_prompt, build_scenario_planning_prompt

logger = logging.getLogger(__name__)


class GroqLLM:
    """Production-grade asynchronous interface for Groq engine client operations."""

    def __init__(self, config: Optional[GroqLLMConfig] = None):
        self.config = config or GroqLLMConfig()

        if not self.config.validate():
            logger.warning("GroqLLM initialized without valid API key - calls will fail")

        self.client = AsyncGroq(api_key=self.config.api_key)
        logger.info(
            "groq_llm_client_initialized", 
            model=self.config.model, 
            temperature=self.config.temperature
        )

    async def generate(
        self, 
        prompt: str, 
        include_system: bool = True,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Generates a text completion from Groq API with defensive timeout 
        handling and automatic retry.
        """
        logger.debug(
            "groq_inference_request_sent", 
            prompt_len=len(prompt), 
            include_system=include_system
        )

        messages = []
        if include_system and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(max_retries):
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
                last_error = exc
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"groq_api_call_failed_attempt_{attempt+1}/{max_retries}", 
                    error=str(exc),
                    wait_seconds=wait_time
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error("groq_api_call_failed_all_retries_exhausted", error=str(last_error))
        raise RuntimeError(
            f"Error connecting to Groq API after {max_retries} attempts: {str(last_error)}"
        )

    async def generate_with_context(
        self, 
        query: str, 
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Assembles a high-density grounded context block to answer 
        the compliance query.
        """
        default_system = (
            "You are an aviation regulatory expert. Provide factual, "
            "well-cited answers based strictly on the provided context."
        )

        structured_prompt = (
            f"Based on the following verified legal context, answer the user's "
            f"aviation compliance question.\n\n"
            f"LEGAL CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Remember to strictly apply constraints, compare FAA/EASA if relevant, "
            f"and remain entirely factual.\n"
            f"ANSWER:"
        )
        return await self.generate(
            structured_prompt, 
            include_system=True, 
            system_prompt=system_prompt or default_system
        )

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classifies the incoming operational request into discrete 
        aviation categories.
        """
        system_prompt = (
            "You are an aviation query classifier. Analyze the query and "
            "return ONLY a JSON object with 'category' and 'confidence' fields."
        )

        classification_prompt = (
            "Classify this UAV operational query into one of these categories: "
            "EMERGENCY_RESPONSE, ADVERSE_WEATHER, REGULATORY_COMPLIANCE, "
            "TECHNICAL_DEGRADATION, SURVEY_MISSION, DELIVERY_MISSION, "
            "INSPECTION_MISSION, GENERAL_OPERATION.\n\n"
            f"Query: {query}\n\n"
            + 'Return JSON: {"category": "CATEGORY_NAME", "confidence": 0.95}'
        )

        try:
            response_text = await self.generate(
                classification_prompt, 
                include_system=True,
                system_prompt=system_prompt,
                max_retries=2
            )

            import json
            # Extract JSON from response
            try:
                result = json.loads(response_text)
                category = result.get("category", "GENERAL_OPERATION")
                confidence = float(result.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError):
                # Fallback: parse manually
                category = "GENERAL_OPERATION"
                confidence = 0.5
                for line in response_text.split("\n"):
                    line = line.strip()
                    if "category" in line.lower():
                        parts = line.split(":")
                        if len(parts) > 1:
                            category = parts[1].strip().strip('"').strip(",")
                    if "confidence" in line.lower():
                        parts = line.split(":")
                        if len(parts) > 1:
                            try:
                                confidence = float(parts[1].strip().strip(","))
                            except ValueError:
                                pass

            logger.info("query_classification_concluded", category=category, confidence=confidence)
            return {"category": category, "confidence": confidence}

        except Exception as exc:
            logger.error("query_classification_failed", error=str(exc))
            return {"category": "GENERAL_OPERATION", "confidence": 0.0}

    async def generate_hypothetical_answer(
        self, 
        query: str,
        features: Optional[Dict[str, Any]] = None,
        scenario_type: str = "general"
    ) -> str:
        """
        Generates an idealized regulatory passage to feed the HyDE search module.
        Uses prompts_v3.build_hyde_prompt for consistency.
        """
        features = features or {}
        prompt = build_hyde_prompt(query, features, scenario_type)

        system_prompt = (
            "You are an aviation regulatory document generator. "
            "Produce factual, regulation-style text."
        )

        return await self.generate(
            prompt, 
            include_system=True,
            system_prompt=system_prompt,
            max_retries=2
        )

    async def analyze_scenario(
        self,
        features: Dict[str, Any],
        free_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze scenario and determine retrieval strategy.
        Uses prompts_v3.build_scenario_planning_prompt.
        """
        prompt = build_scenario_planning_prompt(features, free_text)

        system_prompt = (
            "You are an aviation scenario analyzer. Return ONLY valid JSON."
        )

        try:
            response = await self.generate(
                prompt,
                include_system=True,
                system_prompt=system_prompt,
                max_retries=2
            )

            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("scenario_analysis_json_parse_failed", raw=response[:200])
                return {
                    "scenario_type": "general_operation",
                    "complexity": 0.5,
                    "priority_features": [],
                    "retrieval_strategy": "hybrid",
                    "recommended_top_k": 10,
                    "risk_indicators": []
                }
        except Exception as exc:
            logger.error("scenario_analysis_failed", error=str(exc))
            return {
                "scenario_type": "general_operation",
                "complexity": 0.5,
                "priority_features": [],
                "retrieval_strategy": "hybrid",
                "recommended_top_k": 10,
                "risk_indicators": []
            }


# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# Core LLM gateway driver interacting with external API clusters.
# Dependencies: src/uav_risk/stage2/rag/config_v3.py -> GroqLLMConfig
#               src/uav_risk/stage2/rag/prompts_v3.py -> Prompt Builders
# Dependent Files: src/uav_risk/stage2/rag/hybrid_retriever.py, rag_core_v3.py
# =====================================================================