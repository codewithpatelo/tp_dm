"""Panel EDA: render markdown existente + recompute interactivo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ..data_loader import DataRegistry
from ..parsers import ROOT


EDA_FILES = [
    ROOT / "entregas" / "entrega_3" / "eda_features_v4.md",
    ROOT / "entregas" / "entrega_3" / "error_analysis_v4.md",
    ROOT / "entregas" / "entrega_2" / "eda_v4.md",
]


def render():
    st.caption("Informes guardados en el repo (Markdown) y un gráfico interactivo rápido.")
    st.subheader("Análisis EDA / errores")
    existing = [p for p in EDA_FILES if p.exists()]
    if existing:
        tab_titles = [p.relative_to(ROOT).as_posix() for p in existing] + ["Interactivo"]
    else:
        tab_titles = ["Interactivo"]
    tabs = st.tabs(tab_titles)

    for tab, path in zip(tabs[:-1] if existing else [], existing):
        with tab:
            st.markdown(path.read_text(encoding="utf-8", errors="ignore"))

    with tabs[-1]:
        _render_interactive()


def _render_interactive():
    reg = DataRegistry.get()
    df = reg.table("train_filtered")

    cols = st.columns(3)
    target = cols[0].selectbox("Variable Y", ["price"] + [c for c in df.columns if c != "price"])
    candidates = [c for c in df.select_dtypes(include="number").columns if c != target][:50]
    feat = cols[1].selectbox("Variable X", candidates if candidates else df.columns.tolist())
    method = cols[2].selectbox("Tipo de gráfico", ["scatter", "box por barrio", "histograma X"])

    sub = df[[target, feat] + (["location_2"] if "location_2" in df.columns else [])].dropna()
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=42)

    if method == "scatter":
        fig = px.scatter(sub, x=feat, y=target, opacity=0.4,
                         title=f"{target} vs {feat}")
        st.plotly_chart(fig, use_container_width=True)
    elif method == "box por barrio" and "location_2" in sub.columns:
        top = sub["location_2"].value_counts().head(15).index
        small = sub[sub["location_2"].isin(top)]
        fig = px.box(small, x="location_2", y=target,
                     title=f"{target} por barrio (top 15 más frecuentes)")
        st.plotly_chart(fig, use_container_width=True)
    elif method == "histograma X":
        fig = px.histogram(sub, x=feat, nbins=40, title=f"Distribución de {feat}")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Stats descriptivos del subset"):
        st.dataframe(sub.describe().T, use_container_width=True)
