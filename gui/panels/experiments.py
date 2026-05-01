"""Panel Experiments: tabla unificada del leaderboard + drill-down JSON."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..parsers import list_solutions, load_solution, parse_leaderboard


def render():
    st.subheader("Tabla de resultados (leaderboard local)")
    rows = parse_leaderboard()
    if not rows:
        st.info("No hay entradas en el leaderboard.")
        return
    df = pd.DataFrame([r.__dict__ for r in rows]).drop(columns=["raw"], errors="ignore")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Detalle JSON (soluciones exportadas)")
    sols = list_solutions()
    if not sols:
        st.caption("No hay `solucion-*.json` bajo `entregas/`.")
        return
    sel = st.selectbox("Archivo", [str(p.name) for p in sols])
    chosen = next(p for p in sols if p.name == sel)
    try:
        data = load_solution(chosen)
    except Exception as e:  # noqa: BLE001
        st.error(f"No pude leer {chosen.name}: {e}")
        return
    st.json(data, expanded=False)
