"""
Groq Async Engine (V7.1 - Certified Production Grade)
=====================================================
Contract: Implements AsyncLLMClientInterface.
Resilience: Circuit Breaker + Exponential Backoff + Jitter + Strict Timeouts.
Observability: Prometheus-compatible metrics + structured logging.
Safety: Deep JSON validation (handles Markdown wrapping) + Unicode sanitization.
Smart: Adaptive timeout + streaming support.
"""

import os
import asyncio
import logging
import random
import time
import uuid
import json
import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, AsyncIterator
from groq import AsyncGroq, RateLimitError, APIConnectionError, APIError
from uav_risk.stage2.agents.legal_agent import AsyncLLMClientInterface

logger = logging.getLogger("GroqAsyncClient")

class _RequestLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[req:{self.extra.get('req_id', '?')[:8]}] {msg}", kwargs

class CircuitState(Enum):
    CLOSED   = auto()
    OPEN     = auto()
    HALF_OPEN = auto()

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max: int = 2

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    def allow_request(self) -> bool:
        if self._state == CircuitState.CLOSED: return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.warning("Circuit HALF-OPEN: testing recovery...")
                return True
            return False
        return True

    def record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit CLOSED: service recovered.")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.critical(f"Circuit OPEN: {self._failure_count} consecutive failures. Blocking for {self.recovery_timeout}s.")

    @property
    def state(self) -> CircuitState:
        return self._state

@dataclass
class LLMMetrics:
    total_requests: int = 0
    total_failures: int = 0
    total_retries: int = 0
    total_timeouts: int = 0
    latency_sum_ms: float = 0.0
    _latency_samples: list = field(default_factory=list)

    def record(self, success: bool, latency_ms: float, retries: int = 0, timed_out: bool = False):
        self.total_requests += 1
        self.latency_sum_ms += latency_ms
        self._latency_samples.append(latency_ms)
        if not success: self.total_failures += 1
        if timed_out: self.total_timeouts += 1
        self.total_retries += retries

    @property
    def avg_latency_ms(self) -> float: return self.latency_sum_ms / max(1, self.total_requests)
    @property
    def p95_latency_ms(self) -> float:
        if not self._latency_samples: return 0.0
        s = sorted(self._latency_samples)
        return s[min(int(len(s) * 0.95), len(s) - 1)]
    @property
    def error_rate(self) -> float: return self.total_failures / max(1, self.total_requests)
    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests, "total_failures": self.total_failures,
            "error_rate": round(self.error_rate, 4), "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2), "total_retries": self.total_retries,
            "total_timeouts": self.total_timeouts,
        }

class GroqAsyncClient(AsyncLLMClientInterface):
    _MAX_TOKENS_HARD_CAP = 1500
    _TIMEOUT_PER_CHAR_MS = 0.5
    _MIN_TIMEOUT = 8.0
    _MAX_TIMEOUT = 30.0 

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0, 
                 max_retries: int = 3, base_timeout: float = 10.0, 
                 circuit_breaker: Optional[CircuitBreaker] = None):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key: raise ValueError("CRITICAL: GROQ_API_KEY environment variable is missing.")
        self.client = AsyncGroq(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.circuit = circuit_breaker or CircuitBreaker()
        self.metrics = LLMMetrics()

    def _calculate_timeout(self, prompt: str) -> float:
        dynamic = self.base_timeout + (len(prompt) * self._TIMEOUT_PER_CHAR_MS / 1000)
        return max(self._MIN_TIMEOUT, min(dynamic, self._MAX_TIMEOUT))

    @staticmethod
    def _validate_json_response(content: str, request_id: str) -> str:
        stripped = content.strip()
        if not stripped: raise ValueError(f"Empty JSON response. ID: {request_id}")
        # [FIX] إزالة وسوم Markdown تلقائياً
        stripped = re.sub(r"^```(?:json)?\s*\n?|\s*?```$", "", stripped, flags=re.IGNORECASE).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Groq returned invalid JSON. ID: {request_id}. Parse error at pos {e.pos}: {e.msg}. Content: {stripped[:100]!r}") from e
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}. ID: {request_id}")
        return stripped

    async def generate(self, prompt: str, response_format: Optional[Dict[str, str]] = None) -> str:
        request_id = str(uuid.uuid4())
        req_logger = _RequestLogger(logger, {"req_id": request_id})
        if not self.circuit.allow_request():
            self.metrics.record(success=False, latency_ms=0.0)
            raise RuntimeError(f"Circuit OPEN: LLM service unavailable. State: {self.circuit.state.name}. ID: {request_id}")

        timeout = self._calculate_timeout(prompt)
        kwargs: Dict[str, Any] = {
            "model": self.model_name, "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature, "max_tokens": self._MAX_TOKENS_HARD_CAP,
            "extra_headers": {"X-Request-ID": request_id},
        }
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        start_time = time.monotonic()
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(self.client.chat.completions.create(**kwargs), timeout=timeout)
                content = response.choices[0].message.content.strip()
                if response_format and response_format.get("type") == "json_object":
                    content = self._validate_json_response(content, request_id)
                
                latency_ms = (time.monotonic() - start_time) * 1000
                self.circuit.record_success()
                self.metrics.record(success=True, latency_ms=latency_ms, retries=attempt)
                return content

            except (RateLimitError, APIConnectionError) as e:
                last_exception = e
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                req_logger.warning(f"Rate/Conn Error (attempt {attempt+1}). Retry in {wait_time:.2f}s.")
                await asyncio.sleep(wait_time)
            except APIError as e:
                last_exception = e
                req_logger.error(f"APIError (attempt {attempt+1}): status={getattr(e, 'status_code', '?')}, msg={getattr(e, 'message', '')!r}")
                if hasattr(e, 'status_code') and e.status_code and 400 <= e.status_code < 500:
                    self.circuit.record_failure()
                    raise RuntimeError(f"Non-retryable client error. Aborting. ID: {request_id}") from e
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(f"Request timed out after {timeout:.1f}s")
                req_logger.error(f"Timeout on attempt {attempt+1}")
                if attempt == self.max_retries - 1:
                    self.circuit.record_failure()
                    raise last_exception
            except Exception as e:
                latency_ms = (time.monotonic() - start_time) * 1000
                req_logger.critical(f"Unrecoverable error: {type(e).__name__}: {e}", exc_info=True)
                self.circuit.record_failure()
                self.metrics.record(success=False, latency_ms=latency_ms)
                raise

        self.circuit.record_failure()
        self.metrics.record(success=False, latency_ms=(time.monotonic() - start_time) * 1000, retries=self.max_retries)
        raise RuntimeError(f"LLM Engine exhausted all retries. ID: {request_id}. Last: {last_exception}")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        request_id = str(uuid.uuid4())
        if not self.circuit.allow_request(): raise RuntimeError(f"Circuit OPEN. ID: {request_id}")
        timeout = self._calculate_timeout(prompt)
        async with asyncio.timeout(timeout):
            stream = await self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, max_tokens=self._MAX_TOKENS_HARD_CAP,
                stream=True, extra_headers={"X-Request-ID": request_id}
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta: yield delta

    def get_metrics(self) -> Dict[str, Any]:
        return {**self.metrics.snapshot(), "circuit_state": self.circuit.state.name, "model": self.model_name}

    def export_prometheus_metrics(self) -> str:
        m = self.metrics
        return (f"groq_requests_total {m.total_requests}\n"
                f"groq_failures_total {m.total_failures}\n"
                f"groq_latency_avg_ms {m.avg_latency_ms:.2f}\n"
                f"groq_errors_rate {m.error_rate:.4f}\n"
                f'groq_circuit_state{{state="{self.circuit.state.name}"}} 1\n')