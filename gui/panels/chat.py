"""Panel del asistente: chat, paso actual, métricas mínimas.

El feedback es de **un solo paso a la vez**: mostramos la fase activa con
icono y texto humano (sin timeline ni JSON visible). Los artifacts del
sandbox se publican en el canvas central a través de `st.session_state`.
"""
from __future__ import annotations

import json

import streamlit as st

from ..cognitive import CognitiveAgent
from ..events import (
    EVT_PROPOSAL_PENDING,
    EVT_SKILL_CALLED,
    EVT_SKILL_RESULT,
    EVT_TOKENS,
    get_bus,
)


def _ensure_state() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = CognitiveAgent()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "trace" not in st.session_state:
        st.session_state.trace = []
        bus = get_bus()
        bus.subscribe(
            "*",
            lambda e: st.session_state.trace.append(
                {"type": e.type, "ts": e.ts, "payload": e.payload, "turn_id": e.turn_id}
            ),
        )
    if "proposals" not in st.session_state:
        st.session_state.proposals = []
        get_bus().subscribe(
            EVT_PROPOSAL_PENDING,
            lambda e: st.session_state.proposals.append(e.payload),
        )
    if "tokens_total" not in st.session_state:
        st.session_state.tokens_total = {"in": 0, "out": 0}
        get_bus().subscribe(EVT_TOKENS, _accum_tokens)
    if "last_phase_log" not in st.session_state:
        st.session_state.last_phase_log = []
    if "agent_artifacts" not in st.session_state:
        st.session_state.agent_artifacts = []


def _accum_tokens(e) -> None:
    st.session_state.tokens_total["in"] += e.payload.get("in", 0)
    st.session_state.tokens_total["out"] += e.payload.get("out", 0)


def _phase_line(p: dict) -> str:
    icon = p.get("icon", "•")
    label = p.get("label", p.get("key", "?"))
    return f"{icon} {label}"


def render() -> None:
    _ensure_state()

    with st.expander("Métricas", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("IN", f"{st.session_state.tokens_total['in']:,}")
        m2.metric("OUT", f"{st.session_state.tokens_total['out']:,}")
        m3.metric("Msgs", len(st.session_state.history))

    if st.session_state.last_phase_log:
        with st.expander("Detalle del último turno", expanded=False):
            for p in st.session_state.last_phase_log:
                st.markdown("- " + _phase_line(p))

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.container(key="chat_input_bar"):
        with st.form("agent_chat_form", clear_on_submit=True):
            user_msg = st.text_area(
                "Tu pregunta",
                placeholder="Preguntale a tus datos...",
                height=84,
                label_visibility="collapsed",
            )
            sent = st.form_submit_button("Enviar", width="stretch")

    user_msg = (user_msg or "").strip() if sent else ""
    if user_msg:
        st.session_state.history.append({"role": "user", "content": user_msg})

        current_step = st.empty()
        current_step.info("🧠 Entendiendo tu pregunta…")

        def cb(rec: dict) -> None:
            try:
                current_step.info(_phase_line(rec))
            except Exception:  # noqa: BLE001
                pass

        try:
            wm = st.session_state.agent.turn(user_msg, on_phase=cb)
        except Exception as e:  # noqa: BLE001
            current_step.error(f"⚠️ Error: {e}")
            st.session_state.history.append(
                {"role": "assistant",
                 "content": f"Hubo un error al procesar tu mensaje: `{e}`"},
            )
            st.rerun()
            return

        st.session_state.last_phase_log = list(getattr(wm, "phase_log", []))
        artifacts = list(getattr(wm, "artifacts", []) or [])
        if artifacts:
            st.session_state.agent_artifacts = artifacts

        try:
            current_step.success("✅ Listo")
        except Exception:  # noqa: BLE001
            pass

        st.session_state.history.append(
            {"role": "assistant", "content": wm.answer, "turn_id": wm.turn_id}
        )
        st.rerun()

    with st.expander("Traza técnica (debug)", expanded=False):
        recent = st.session_state.trace[-25:]
        for ev in reversed(recent):
            t = ev["type"]
            if t in {EVT_SKILL_CALLED, EVT_SKILL_RESULT}:
                st.code(f"{t}: {json.dumps(ev.get('payload'), default=str)[:300]}",
                        language="json")
            else:
                st.caption(f"`{t}`")
