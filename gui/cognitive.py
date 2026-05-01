"""Agente Code-First.

Flujo simple:

1. **Generar código**: 1 LLM call con `codegen_system_prompt`. El LLM
   devuelve JSON con `analysis_code` o, si la pregunta no requiere datos,
   con `answer` directo.
2. **Ejecutar código**: corre en sandbox read-only sobre `train_filtered`
   y `test`. Captura stdout, outputs y artifacts publicados.
3. **(Opcional) Reintentar 1 vez**: si el código falló por error humano
   (columna inexistente, etc.), pedimos al LLM una corrección con el
   traceback y volvemos a ejecutar.
4. **Sintetizar respuesta**: 1 LLM call con `synth_system_prompt` y los
   datos reales (outputs/stdout/artifacts) → markdown final para el chat.

UI feedback: emitimos fases concisas con icono + texto humano. La traza
detallada queda en `wm.phase_log` y eventos `EVT_COGNITIVE_PHASE`.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .data_loader import DataRegistry
from .events import (
    EVT_COGNITIVE_PHASE,
    EVT_ERROR,
    EVT_REPORT,
    EVT_SKILL_CALLED,
    EVT_SKILL_RESULT,
    EVT_TOKENS,
    EVT_USER_MESSAGE,
    get_bus,
)
from .llm import call_llm_json
from .memory import EpisodicMemory, WorkingMemory
from .sandbox import AnalysisResult, execute_analysis
from .system_prompt import codegen_system_prompt, synth_system_prompt, CODEGEN_FIX_HINT


MAX_FIX_ATTEMPTS = 1
SANDBOX_TIMEOUT_S = 45.0


# ---------- iconografía minimalista ----------------------------------------

ICON_PHASE: dict[str, tuple[str, str]] = {
    "thinking":   ("🧠", "Entendiendo tu pregunta"),
    "generating": ("✏️", "Generando análisis"),
    "executing":  ("⚙️", "Ejecutando código"),
    "fixing":     ("🔁", "Corrigiendo el análisis"),
    "rendering":  ("🎨", "Armando visualización"),
    "answering":  ("📝", "Redactando respuesta"),
    "done":       ("✅", "Listo"),
    "error":      ("⚠️", "Hubo un error"),
}


def phase_icon(key: str) -> tuple[str, str]:
    return ICON_PHASE.get(key, ("•", key))


# ---------- agente ---------------------------------------------------------


class CognitiveAgent:
    """Agente Code-First (mantiene el nombre histórico para no romper imports)."""

    def __init__(self) -> None:
        self.episodic = EpisodicMemory()
        self.bus = get_bus()

    # ---------- API pública ------------------------------------------------

    def turn(
        self,
        user_message: str,
        context_history: list[dict] | None = None,
        on_phase: Callable[[dict], None] | None = None,
    ) -> WorkingMemory:
        wm = WorkingMemory(user_message=user_message)
        wm.artifacts = []  # type: ignore[attr-defined]
        self.bus.emit(EVT_USER_MESSAGE, {"text": user_message}, turn_id=wm.turn_id)

        try:
            self._phase(wm, on_phase, "thinking")

            datasets = self._load_datasets()

            # ---- 1) generar código (o respuesta directa) ----
            self._phase(wm, on_phase, "generating")
            plan_obj = self._codegen(wm, user_message, prior=None)

            if "answer" in plan_obj and plan_obj.get("answer"):
                wm.answer = str(plan_obj["answer"]).strip()
                self._phase(wm, on_phase, "answering")
                self.bus.emit(EVT_REPORT, {"text": wm.answer}, turn_id=wm.turn_id)
                self._phase(wm, on_phase, "done")
                self.episodic.save_turn(wm)
                return wm

            code = (plan_obj.get("analysis_code") or "").strip()
            plan_text = (plan_obj.get("plan") or "").strip()
            assumptions = (plan_obj.get("assumptions") or "").strip()
            wm.intent = {
                "plan": plan_text,
                "assumptions": assumptions,
                "expected_artifact": plan_obj.get("expected_artifact"),
            }

            if not code:
                # Sin código y sin answer → degradación amable.
                wm.answer = (
                    "No pude armar el análisis para esa pregunta esta vez. "
                    "¿Podés reformularla con un poco más de contexto "
                    "(barrio, tipo de propiedad, métrica)?"
                )
                self._phase(wm, on_phase, "answering")
                self.bus.emit(EVT_REPORT, {"text": wm.answer}, turn_id=wm.turn_id)
                self._phase(wm, on_phase, "done")
                self.episodic.save_turn(wm)
                return wm

            # ---- 2) ejecutar ----
            self._phase(wm, on_phase, "executing", extra={"label": _exec_label(plan_text)})
            self.bus.emit(EVT_SKILL_CALLED,
                          {"skill": "execute_analysis", "args": {"code_chars": len(code)}},
                          turn_id=wm.turn_id)
            result = execute_analysis(code, datasets, timeout_s=SANDBOX_TIMEOUT_S)
            wm.add_observation(
                "execute_analysis",
                {"code": _short(code, 600)},
                _result_summary(result),
                ok=result.ok,
                error=(result.error or None) if not result.ok else None,
            )
            self.bus.emit(EVT_SKILL_RESULT,
                          {"skill": "execute_analysis", "ok": result.ok,
                           "n_artifacts": len(result.artifacts)},
                          turn_id=wm.turn_id)

            # ---- 3) reintento si falló ----
            if not result.ok:
                for attempt in range(MAX_FIX_ATTEMPTS):
                    self._phase(wm, on_phase, "fixing",
                                extra={"label": "Ajustando el análisis"})
                    plan_obj2 = self._codegen(
                        wm, user_message,
                        prior={"code": code, "error": result.error,
                               "traceback": result.traceback,
                               "stdout": result.stdout},
                    )
                    if "answer" in plan_obj2 and plan_obj2.get("answer"):
                        wm.answer = str(plan_obj2["answer"]).strip()
                        self._phase(wm, on_phase, "answering")
                        self.bus.emit(EVT_REPORT, {"text": wm.answer}, turn_id=wm.turn_id)
                        self._phase(wm, on_phase, "done")
                        self.episodic.save_turn(wm)
                        return wm
                    code2 = (plan_obj2.get("analysis_code") or "").strip()
                    if not code2:
                        break
                    self._phase(wm, on_phase, "executing",
                                extra={"label": "Reintentando ejecución"})
                    self.bus.emit(EVT_SKILL_CALLED,
                                  {"skill": "execute_analysis",
                                   "args": {"code_chars": len(code2), "attempt": attempt + 2}},
                                  turn_id=wm.turn_id)
                    result = execute_analysis(code2, datasets, timeout_s=SANDBOX_TIMEOUT_S)
                    wm.add_observation(
                        "execute_analysis",
                        {"code": _short(code2, 600), "attempt": attempt + 2},
                        _result_summary(result),
                        ok=result.ok,
                        error=(result.error or None) if not result.ok else None,
                    )
                    self.bus.emit(EVT_SKILL_RESULT,
                                  {"skill": "execute_analysis", "ok": result.ok,
                                   "n_artifacts": len(result.artifacts)},
                                  turn_id=wm.turn_id)
                    code = code2
                    if result.ok:
                        break

            # ---- 4) artifacts → wm para que la UI los publique ----
            wm.artifacts = list(result.artifacts)  # type: ignore[attr-defined]
            if result.artifacts:
                self._phase(wm, on_phase, "rendering",
                            extra={"label": f"Publicando {len(result.artifacts)} artifact(s)"})

            # ---- 5) sintetizar ----
            self._phase(wm, on_phase, "answering")
            wm.answer = self._synth(wm, user_message, plan_text, code, result)
            self.bus.emit(EVT_REPORT, {"text": wm.answer}, turn_id=wm.turn_id)
            self._phase(wm, on_phase, "done")

        except Exception as e:  # noqa: BLE001
            self._phase(wm, on_phase, "error", extra={"label": str(e)[:140]})
            self.bus.emit(EVT_ERROR, {"error": str(e)}, turn_id=wm.turn_id)
            wm.answer = (
                f"**Hubo un error inesperado al procesar tu mensaje:** `{e}`\n\n"
                "Probá reformular o avisame para depurarlo."
            )

        self.episodic.save_turn(wm)
        return wm

    # ---------- helpers ----------------------------------------------------

    def _load_datasets(self) -> dict:
        reg = DataRegistry.get()
        return {
            "train_filtered": reg.table("train_filtered"),
            "test": reg.table("test"),
        }

    def _codegen(
        self,
        wm: WorkingMemory,
        user_message: str,
        prior: dict | None,
    ) -> dict:
        sys = codegen_system_prompt()
        user_block = _build_codegen_user(user_message, prior)
        obj, resp = call_llm_json(
            sys,
            user_block,
            max_tokens=2500,
            temperature=0.1,
            fallback={"answer": ""},
        )
        wm.tokens_in += resp.tokens_in
        wm.tokens_out += resp.tokens_out
        self.bus.emit(EVT_TOKENS,
                      {"in": resp.tokens_in, "out": resp.tokens_out, "stage": "codegen"},
                      turn_id=wm.turn_id)
        if not isinstance(obj, dict):
            obj = {"answer": ""}
        return obj

    def _synth(
        self,
        wm: WorkingMemory,
        user_message: str,
        plan_text: str,
        code: str,
        result: AnalysisResult,
    ) -> str:
        sys = synth_system_prompt()
        user_block = _build_synth_user(user_message, plan_text, code, result)
        obj, resp = call_llm_json(
            sys,
            user_block,
            max_tokens=1800,
            temperature=0.2,
            fallback={"answer": ""},
        )
        wm.tokens_in += resp.tokens_in
        wm.tokens_out += resp.tokens_out
        self.bus.emit(EVT_TOKENS,
                      {"in": resp.tokens_in, "out": resp.tokens_out, "stage": "synth"},
                      turn_id=wm.turn_id)

        answer = ""
        if isinstance(obj, dict):
            answer = str(obj.get("answer") or "").strip()
            followups = obj.get("followups") or []
            if isinstance(followups, list) and followups:
                bullets = "\n".join(f"- {str(f)[:140]}" for f in followups[:3])
                if bullets and bullets not in answer:
                    answer = answer.rstrip() + f"\n\n**Próximas preguntas:**\n{bullets}"
        if answer:
            return answer

        # Fallback: armamos una respuesta mínima usando lo que sí tenemos.
        return _fallback_answer_from_result(result, plan_text)

    def _phase(
        self,
        wm: WorkingMemory,
        on_phase: Callable[[dict], None] | None,
        key: str,
        extra: dict | None = None,
    ) -> None:
        icon, default_label = phase_icon(key)
        rec = {
            "key": key,
            "icon": icon,
            "label": (extra or {}).get("label") or default_label,
            "ts": time.time(),
        }
        if extra:
            for k, v in extra.items():
                if k not in rec:
                    rec[k] = v
        wm.phase_log.append(rec)
        self.bus.emit(EVT_COGNITIVE_PHASE, rec, turn_id=wm.turn_id)
        if on_phase:
            try:
                on_phase(rec)
            except Exception:  # noqa: BLE001
                pass


# ---------- builders de prompts del usuario --------------------------------


def _build_codegen_user(user_message: str, prior: dict | None) -> str:
    parts = [f"Pregunta del usuario:\n```\n{user_message.strip()}\n```"]
    if prior:
        parts.append(CODEGEN_FIX_HINT)
        parts.append("Código previo:\n```python\n" + (prior.get("code") or "")[:2500] + "\n```")
        if prior.get("error"):
            parts.append("Error:\n```\n" + str(prior["error"])[:600] + "\n```")
        if prior.get("traceback"):
            parts.append("Traceback:\n```\n" + str(prior["traceback"])[-1200:] + "\n```")
        if prior.get("stdout"):
            parts.append("Stdout previo:\n```\n" + str(prior["stdout"])[-600:] + "\n```")
    parts.append(
        "Devolvé el JSON con `plan`+`analysis_code`+`expected_artifact` o "
        "con `answer` si no hace falta análisis."
    )
    return "\n\n".join(parts)


def _build_synth_user(
    user_message: str,
    plan_text: str,
    code: str,
    result: AnalysisResult,
) -> str:
    parts = [f"Pregunta del usuario:\n```\n{user_message.strip()}\n```"]
    if plan_text:
        parts.append(f"Plan declarado:\n{plan_text}")
    parts.append("Código ejecutado:\n```python\n" + _short(code, 1800) + "\n```")
    if result.ok:
        parts.append("Resultado de la ejecución:\n```\n" + result.to_summary() + "\n```")
    else:
        parts.append("La ejecución FALLÓ. Detalles:\n```\n" + result.to_summary() + "\n```")
    parts.append(
        "Devolvé un JSON `{\"answer\": \"...markdown...\", \"followups\": [...] }`. "
        "Hablá en español rioplatense, sé breve y honesto."
    )
    return "\n\n".join(parts)


# ---------- helpers de resumen --------------------------------------------


def _short(s: str, n: int = 600) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."


def _result_summary(result: AnalysisResult) -> dict:
    return {
        "ok": result.ok,
        "elapsed_s": round(result.elapsed_s, 2),
        "n_artifacts": len(result.artifacts),
        "stdout_tail": (result.stdout or "")[-600:],
        "outputs_keys": list(result.outputs.keys())[:12],
        "error": result.error or None,
    }


def _exec_label(plan_text: str) -> str:
    if not plan_text:
        return "Ejecutando código de análisis"
    return f"Ejecutando: {plan_text[:80]}"


def _fallback_answer_from_result(result: AnalysisResult, plan_text: str) -> str:
    """Si el LLM de síntesis no devolvió nada útil, armamos algo razonable."""
    if not result.ok:
        return (
            "No pude completar el análisis esta vez. "
            f"El código falló con: `{(result.error or '')[:200]}`. "
            "Probá reformular la pregunta con menos restricciones o avisame "
            "para revisarlo."
        )
    parts: list[str] = []
    if plan_text:
        parts.append(f"Hice esto: _{plan_text}_.")
    if result.stdout.strip():
        parts.append("Notas del análisis:\n```\n" + result.stdout.strip()[-800:] + "\n```")
    if result.artifacts:
        kinds = ", ".join(sorted({a.get("kind", "?") for a in result.artifacts}))
        parts.append(f"Publiqué **{len(result.artifacts)}** artifact(s) ({kinds}) "
                     "en el panel central. Mirálo ahí.")
    if not parts:
        parts.append("Ejecuté el análisis pero no quedó información para resumir. "
                     "Reformulá la pregunta y vuelvo a intentar.")
    return "\n\n".join(parts)


# ---------- compatibilidad: re-export para chat.py legacy ------------------

# El panel viejo importaba `SKILL_UI`/`skill_label`. Mantenemos algo simple
# para no romper, aunque ya no usamos catálogo de skills nombradas.
SKILL_UI: dict[str, tuple[str, str]] = {
    "execute_analysis": ("⚙️", "Ejecutando código"),
}


def skill_label(name: str) -> tuple[str, str]:
    return SKILL_UI.get(name, ("⚙️", name))


# Útil para tests / debugging desde notebook.
def _example_codegen_payload() -> dict:
    return {
        "plan": "ejemplo: comparar mediana de price por location_3",
        "analysis_code": (
            "df = train_filtered.copy()\n"
            "agg = (df.groupby('location_3')['price']\n"
            "         .agg(n='size', mediana='median')\n"
            "         .query('n >= 30')\n"
            "         .sort_values('mediana', ascending=False)\n"
            "         .head(10).reset_index())\n"
            "publish_artifact('table', title='Top 10 barrios por mediana', data=agg)\n"
            "print('Top 10 calculado sobre', len(df), 'filas')\n"
        ),
        "expected_artifact": "tabla con n y mediana por barrio",
        "assumptions": "uso location_3 como barrio; min_n=30",
    }


__all__ = [
    "CognitiveAgent",
    "ICON_PHASE",
    "phase_icon",
    "skill_label",
    "SKILL_UI",
    "MAX_FIX_ATTEMPTS",
    "SANDBOX_TIMEOUT_S",
]


# Silenciar referencia no usada (útil para debugging interactivo).
_ = json
