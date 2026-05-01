"""Panel Status: champion card + últimas runs + plan TODOs + backlog top 5."""
from __future__ import annotations

import streamlit as st

from ..parsers import (
    champion, context_section, parse_leaderboard, parse_plan_todos,
)


def render():
    st.subheader("Resumen de experimentos (local)")

    c = champion()
    if c is None:
        st.info("No se encontró `leaderboard.md` bajo `entregas/`, o está vacío.")
    else:
        cols = st.columns(4)
        cols[0].metric("Mejor run (Kaggle o local)", c.nombre[:32])
        cols[1].metric("RMSE Kaggle", f"{c.rmse_kaggle:,.0f}" if c.rmse_kaggle else "—")
        cols[2].metric("RMSE CV5 (media)", f"{c.rmse_cv5_mean:,.0f}" if c.rmse_cv5_mean else "—")
        cols[3].metric("RMSE holdout temporal", f"{c.rmse_holdout_temporal:,.0f}"
                       if c.rmse_holdout_temporal else "—")

    with st.expander("Últimas 6 corridas", expanded=True):
        rows = parse_leaderboard()
        if not rows:
            st.caption("Sin runs.")
        else:
            import pandas as pd
            df = pd.DataFrame([r.__dict__ for r in rows[-6:][::-1]])
            df = df.drop(columns=["raw"], errors="ignore")
            st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Plan activo (TODOs en ~/.cursor/plans/)", expanded=False):
        todos = parse_plan_todos()
        if not todos:
            st.caption("Sin TODOs activos.")
        else:
            for t in todos[:30]:
                ic = {"completed": "[x]", "in_progress": "[ ]",
                      "pending": "[ ]", "cancelled": "[~]"}.get(t.status, "[ ]")
                st.markdown(f"- {ic} `{t.status}` · {t.content} · _{t.plan_file}_")

    with st.expander("Backlog (extracto de CONTEXT.md)", expanded=False):
        backlog = context_section("Backlog")
        if not backlog:
            st.caption("No hay sección «Backlog» en CONTEXT o el archivo no está.")
        else:
            st.markdown(backlog[:3000])
