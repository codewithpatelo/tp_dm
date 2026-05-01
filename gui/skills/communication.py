"""Skills de comunicación con el usuario: clarify, report (placeholder)."""
from __future__ import annotations

from . import register


@register(
    "clarify",
    "Pregunta al usuario para desambiguar. La respuesta vuelve como nuevo turno.",
    {"question": "string"},
)
def clarify(question: str):
    return {"clarification_question": question, "needs_user_input": True}


@register(
    "report",
    "Marca que se debe generar el reporte final (lo arma el loop cognitivo).",
    {"draft": "string?"},
)
def report(draft: str = ""):
    return {"draft": draft, "ready_for_render": True}
