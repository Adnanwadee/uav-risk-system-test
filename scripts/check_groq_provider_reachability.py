from __future__ import annotations

import asyncio
import json

from uav_risk.stage2.llm.orchestrator import (
    build_llm_orchestrator_from_env,
    load_llm_runtime_config_from_env,
)


async def _run() -> dict:
    cfg = load_llm_runtime_config_from_env()
    orchestrator = build_llm_orchestrator_from_env()
    provider = getattr(orchestrator, "provider", None)

    result = {
        "config_enabled": bool(cfg.enabled),
        "provider": cfg.provider,
        "model_name": cfg.model_name,
        "has_provider": provider is not None,
        "call_attempted": False,
        "call_succeeded": False,
        "error_type": None,
        "error_message_short": None,
    }

    if provider is None:
        return result

    result["call_attempted"] = True
    prompt = (
        "Return exactly this JSON object and nothing else: "
        '{"status":"ok","message":"groq reachable"}'
    )

    try:
        payload = provider.generate_json(prompt, "ReachabilityProbe")
        if hasattr(payload, "__await__"):
            payload = await payload

        if isinstance(payload, dict):
            result["call_succeeded"] = True
        else:
            result["error_type"] = "unexpected_payload"
            result["error_message_short"] = "provider payload was not a JSON object"
    except Exception as exc:
        result["error_type"] = str(getattr(exc, "reason_code", type(exc).__name__) or type(exc).__name__)
        msg = str(getattr(exc, "safe_message", "") or "provider call failed")
        lowered = msg.lower()
        short = "provider call failed"
        if "timeout" in lowered:
            short = "provider timeout"
        elif "auth" in lowered:
            short = "provider auth error"
        elif "rate" in lowered:
            short = "provider rate limit"
        elif "json" in lowered:
            short = "invalid json"
        elif "network" in lowered:
            short = "provider network call error"
        result["error_message_short"] = short

    return result


def main() -> int:
    payload = asyncio.run(_run())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
