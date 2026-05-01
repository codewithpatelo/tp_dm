"""Skills de hipótesis/experimentos. Persisten a semantic memory cuando aplica."""
from __future__ import annotations

from . import register
from ..memory_semantic import SemanticMemory


def _sem() -> SemanticMemory:
    return SemanticMemory.get()


@register(
    "propose_hypothesis",
    "Registra una hipótesis (texto) en memoria semántica con tag 'hypothesis'.",
    {"text": "string", "evidence": "string?"},
)
def propose_hypothesis(text: str, evidence: str = ""):
    rec = _sem().append({"kind": "hypothesis", "text": text, "evidence": evidence})
    return {"id": rec["id"], "stored": True}


@register(
    "propose_experiment",
    "Registra propuesta de experimento (objetivo, diseño, métrica esperada).",
    {"text": "string", "design": "string?", "expected_metric": "string?"},
)
def propose_experiment(text: str, design: str = "", expected_metric: str = ""):
    rec = _sem().append({
        "kind": "experiment", "text": text,
        "design": design, "expected_metric": expected_metric,
    })
    return {"id": rec["id"], "stored": True}


@register(
    "recall_lessons",
    "Trae lecciones (kind=lesson|hypothesis|experiment) que matchean el topic.",
    {"topic": "string", "k": "int default 5"},
)
def recall_lessons(topic: str, k: int = 5):
    return _sem().search(topic, k=k)


@register(
    "update_semantic_memory",
    "Agrega una nota a la memoria semántica (tipo lesson, decision, anti-pattern).",
    {"kind": "lesson|decision|anti-pattern", "text": "string", "tags": "list[str]?"},
)
def update_semantic_memory(kind: str, text: str, tags: list[str] | None = None):
    rec = _sem().append({"kind": kind, "text": text, "tags": tags or []})
    return {"id": rec["id"], "stored": True}


@register(
    "suggest_followup_questions",
    "Genera preguntas de seguimiento (no llama LLM, las arma a partir de heurísticas).",
    {"topic": "string"},
)
def suggest_followup_questions(topic: str):
    return [
        f"¿Querés profundizar en {topic} segmentando por barrio o tipo de propiedad?",
        f"¿Hacemos un check de drift train→test sobre las features ligadas a {topic}?",
        f"¿Lo dejamos como hipótesis para validar con un experimento controlado?",
    ]
