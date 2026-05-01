"""Wrapper de LLM: OpenAI (Chat Completions) o Anthropic (Messages API).

Estructurado: pedimos JSON en el prompt y validamos. Sin Tool Use.

OpenAI Chat Completions: modelos recientes (p. ej. gpt-5, o1, o3) requieren
`max_completion_tokens` en lugar de `max_tokens`. Ver:
https://platform.openai.com/docs/api-reference/chat/create
(parámetros `max_tokens` vs `max_completion_tokens` según familia de modelo)

Proveedor: `GUI_LLM_PROVIDER=openai|anthropic` o autodetección.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:  # noqa: BLE001
    OpenAI = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic
except ImportError:  # noqa: BLE001
    Anthropic = None  # type: ignore[assignment]

Provider = Literal["openai", "anthropic"]


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    raw: Any = None
    provider: str = "openai"


# ---------- keys & provider ---------------------------------------------------

OPENAI_KEY_VARS = ("OPENAI_API_KEY", "OPENAI_KEY")

ANTHROPIC_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTROPHIC_API_KEY",
    "ANTHROPIC_KEY",
    "CLAUDE_API_KEY",
)


def _get_openai_key() -> str | None:
    for name in OPENAI_KEY_VARS:
        v = os.environ.get(name)
        if v:
            return v
    return None


def _get_anthropic_key() -> str | None:
    for name in ANTHROPIC_KEY_VARS:
        v = os.environ.get(name)
        if v:
            return v
    return None


def resolve_provider() -> Provider:
    """Elige proveedor. Por defecto: openai si hay key, si no anthropic."""
    explicit = os.environ.get("GUI_LLM_PROVIDER", "").strip().lower()
    if explicit in ("openai", "anthropic"):
        return explicit  # type: ignore[return-value]
    if _get_openai_key():
        return "openai"
    if _get_anthropic_key():
        return "anthropic"
    return "openai"


# OpenAI: frontier asequible (puede fijar snapshot, ej. gpt-5-mini-2025-08-07)
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
# Anthropic: flagship reciente; override con GUI_LLM_MODEL si bajáis a Haiku, etc.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def default_model() -> str:
    """Modelo default según proveedor (o GUI_LLM_MODEL)."""
    override = os.environ.get("GUI_LLM_MODEL", "").strip()
    p = resolve_provider()
    if override:
        return override
    if p == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_ANTHROPIC_MODEL


# ---------- clientes (lazy) ---------------------------------------------------

_openai_client: Any = None
_anthropic_client: Any = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if OpenAI is None:
            raise RuntimeError("Falta `pip install openai`")
        key = _get_openai_key()
        if not key:
            raise RuntimeError(
                f"Falta API key de OpenAI en .env. Variables: {', '.join(OPENAI_KEY_VARS)}"
            )
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if Anthropic is None:
            raise RuntimeError("Falta `pip install anthropic`")
        key = _get_anthropic_key()
        if not key:
            raise RuntimeError(
                f"Falta API key de Anthropic. Variables: {', '.join(ANTHROPIC_KEY_VARS)}"
            )
        _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


def _openai_uses_max_completion_tokens(model: str) -> bool:
    """Familias que, según la API actual, exigen `max_completion_tokens` (no `max_tokens`)."""
    m = (model or "").lower().strip()
    if m.startswith("gpt-5"):
        return True
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    if m.startswith("gpt-4.1") or m.startswith("computer-use-preview"):
        return True
    return False


def _openai_supports_custom_temperature(model: str) -> bool:
    """Familias que NO admiten `temperature` (sólo el valor default = 1).

    Según docs OpenAI:
    - Razonamiento (o1, o3, o4-...): no temperature.
    - GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano y snapshots): no temperature.
    Ver: https://platform.openai.com/docs/api-reference/chat/create#chat_create-temperature
    y notas en cada modelo (Reasoning token support).
    """
    m = (model or "").lower()
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return False
    if m.startswith("gpt-5"):
        return False
    return True


def _is_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _call_openai(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None = "minimal",
    json_mode: bool = False,
) -> LLMResponse:
    client = get_openai_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Modelos de razonamiento (gpt-5*, o1*, o3*, o4*) consumen tokens internamente.
    # Subimos un "piso" para que la respuesta final no quede vacía.
    if _is_reasoning(model):
        max_tokens = max(max_tokens, 4000)

    def _build_kwargs(
        mct: bool, with_temp: bool, with_reasoning: bool, with_json: bool
    ) -> dict[str, Any]:
        k: dict[str, Any] = {"model": model, "messages": messages}
        if mct:
            k["max_completion_tokens"] = max_tokens
        else:
            k["max_tokens"] = max_tokens
        if with_temp and _openai_supports_custom_temperature(model):
            k["temperature"] = temperature
        if with_reasoning and reasoning_effort and _is_reasoning(model):
            # `reasoning_effort` admitido por gpt-5 y o-series. Ahorra latencia/tokens.
            k["reasoning_effort"] = reasoning_effort
        if with_json:
            k["response_format"] = {"type": "json_object"}
        return k

    mct = _openai_uses_max_completion_tokens(model)
    with_temp = True
    with_reasoning = True
    with_json = json_mode
    last_err: Exception | None = None
    for _ in range(6):
        try:
            completion = client.chat.completions.create(
                **_build_kwargs(mct, with_temp, with_reasoning, with_json)
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            err = str(e).lower()
            if mct is False and (
                "max_completion_tokens" in err
                or ("max_tokens" in err and "not supported" in err)
            ):
                mct = True
                continue
            if mct is True and "max_completion_tokens" in err and "not supported" in err:
                mct = False
                continue
            if "temperature" in err and (
                "not supported" in err
                or "does not support" in err
                or "unsupported value" in err
                or "only the default" in err
            ):
                with_temp = False
                continue
            if "reasoning_effort" in err and (
                "not supported" in err
                or "unsupported" in err
                or "unknown parameter" in err
            ):
                with_reasoning = False
                continue
            if with_json and "response_format" in err:
                with_json = False
                continue
            raise
    else:
        if last_err:
            raise last_err
        raise RuntimeError("OpenAI: sin respuesta")

    text = (completion.choices[0].message.content or "").strip()
    u = completion.usage
    t_in = getattr(u, "prompt_tokens", None) if u else None
    t_out = getattr(u, "completion_tokens", None) if u else None
    return LLMResponse(
        text=text,
        tokens_in=int(t_in or 0),
        tokens_out=int(t_out or 0),
        raw=completion,
        provider="openai",
    )


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    client = get_anthropic_client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        b.text for b in msg.content if hasattr(b, "text")
    )
    return LLMResponse(
        text=text,
        tokens_in=getattr(msg.usage, "input_tokens", 0) or 0,
        tokens_out=getattr(msg.usage, "output_tokens", 0) or 0,
        raw=msg,
        provider="anthropic",
    )


def call_llm(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 1500,
    temperature: float = 0.2,
    provider: Provider | None = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Llamada unificada. `model` default según `GUI_LLM_MODEL` o proveedor."""
    p: Provider = provider or resolve_provider()
    m = model or default_model()
    if p == "openai":
        return _call_openai(system, user, m, max_tokens, temperature, json_mode=json_mode)
    return _call_anthropic(system, user, m, max_tokens, temperature)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def call_llm_json(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 1500,
    temperature: float = 0.0,
    fallback: dict | None = None,
    provider: Provider | None = None,
) -> tuple[Any, LLMResponse]:
    """Igual que call_llm pero parsea JSON. Defensivo: si falla, retry con más tokens."""
    enriched = (
        user
        + "\n\nResponde EXCLUSIVAMENTE con un bloque JSON válido. "
        + "Nada antes ni después."
    )
    resp = call_llm(
        system, enriched, model=model, max_tokens=max_tokens,
        temperature=temperature, provider=provider, json_mode=True,
    )
    txt = (resp.text or "").strip()
    parsed = _try_parse_json(txt) if txt else None

    # Reintento si el modelo devolvió vacío (típico en reasoning models cuando
    # el budget se consumió en thinking). Doblamos el presupuesto.
    if parsed is None and (not txt) and max_tokens < 8000:
        retry_tokens = min(max_tokens * 2, 8000)
        resp2 = call_llm(
            system, enriched, model=model, max_tokens=retry_tokens,
            temperature=temperature, provider=provider, json_mode=True,
        )
        # acumular tokens
        resp.tokens_in += resp2.tokens_in
        resp.tokens_out += resp2.tokens_out
        resp.text = resp2.text
        txt = (resp2.text or "").strip()
        parsed = _try_parse_json(txt) if txt else None

    if parsed is None:
        parsed = fallback if fallback is not None else {}
    return parsed, resp


def _try_parse_json(txt: str) -> Any:
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(txt)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i = txt.find(opener)
        j = txt.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(txt[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None
