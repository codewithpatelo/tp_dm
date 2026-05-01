"""Panel Proposals: muestra cambios propuestos por el agente y permite Apply/Reject."""
from __future__ import annotations

import difflib
from pathlib import Path

import streamlit as st

from ..events import EVT_PROPOSAL_RESOLVED, get_bus
from ..parsers import ROOT


def render():
    st.caption("Cuando el agente propone un cambio a un archivo del repo, revisalo acá (diff y Aplicar / Rechazar).")
    st.subheader("Cambios propuestos")
    proposals = st.session_state.get("proposals", [])
    pending = [p for p in proposals if p.get("status") == "pending"]
    if not pending:
        st.caption("Nada pendiente por ahora.")
        _show_history(proposals)
        return

    for prop in pending:
        with st.expander(f"`{prop['path']}` · {prop['id']}", expanded=True):
            st.markdown(f"**Motivo**: {prop.get('rationale', '—')}")
            diff = "\n".join(difflib.unified_diff(
                prop.get("current", "").splitlines(),
                prop.get("proposed", "").splitlines(),
                fromfile=f"a/{prop['path']}", tofile=f"b/{prop['path']}",
                lineterm="",
            ))
            st.code(diff or "(archivo nuevo o sin diff visible)", language="diff")
            cols = st.columns(2)
            if cols[0].button("Aplicar", key=f"apply_{prop['id']}", type="primary"):
                _apply(prop)
                st.success(f"Aplicado: {prop['path']}")
                st.rerun()
            if cols[1].button("Rechazar", key=f"reject_{prop['id']}"):
                prop["status"] = "rejected"
                get_bus().emit(EVT_PROPOSAL_RESOLVED,
                               {"id": prop["id"], "decision": "rejected"})
                st.warning("Propuesta descartada.")
                st.rerun()

    _show_history(proposals)


def _apply(prop: dict):
    target = ROOT / prop["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(prop["proposed"], encoding="utf-8")
    prop["status"] = "applied"
    get_bus().emit(EVT_PROPOSAL_RESOLVED,
                   {"id": prop["id"], "decision": "applied", "path": prop["path"]})


def _show_history(proposals):
    resolved = [p for p in proposals if p.get("status") in {"applied", "rejected"}]
    if not resolved:
        return
    with st.expander(f"Historial de decisiones ({len(resolved)})", expanded=False):
        for p in resolved[-10:]:
            st.caption(f"`{p['status']}` · {p['path']} · {p['id']}")
