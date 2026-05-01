"""Hablá con tus datos — GUI principal (TP Data Mining, FCEN-UBA).

Layout tipo IDE: navegación izquierda, canvas/datos al centro y agente fijo
a la derecha. No usa el sidebar nativo de Streamlit.

El canvas central renderiza los artifacts publicados por el agente Code-First
(métricas, tablas, gráficos plotly o markdown).

Ejecutar: python -m streamlit run gui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Hablá con tus datos · TP DM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS estable: sin sidebar nativo, header fijo, nav izquierda y agente derecho.
st.markdown(
    """
<style>
    /* Ocultar por completo el sidebar nativo para evitar el panel vacío. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    div.block-container {
        padding-top: 6.25rem !important;  /* toolbar Streamlit + header */
        max-width: 100%;
        padding-left: 160px;              /* reserva nav fija izquierda */
        padding-right: 292px;             /* reserva chat fijo derecho */
        box-sizing: border-box;
    }
    .stMetric { background: rgba(30, 41, 59, 0.4); border-radius: 0.5rem; padding: 0.4rem; }

    /* === Header FIJO (no scrollea) === */
    .app-fixed-header {
        position: fixed;
        top: 2.35rem;                     /* debajo de toolbar Streamlit */
        left: 0;
        right: 0;
        z-index: 999;
        background: rgba(14, 17, 23, 0.94);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 0.45rem 1rem 0.45rem 1rem;
        border-bottom: 1px solid rgba(124, 58, 237, 0.30);
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .app-fixed-header h1 {
        margin: 0; font-size: 1.12rem; line-height: 1.15; color: #e4e4e7;
    }
    .app-fixed-header .subtitle {
        font-size: 0.70rem; opacity: 0.72; margin-top: 0.05rem;
    }

    /* Layout tipo IDE */
    .st-key-nav_panel {
        position: fixed;
        left: 10px;
        top: 6.05rem;
        width: 138px;
        height: calc(100vh - 6.75rem);
        overflow-y: auto;
        padding: 0.65rem;
        box-sizing: border-box;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 0.75rem;
        background: rgba(15, 23, 42, 0.38);
        z-index: 998;
    }
    .st-key-agent_panel {
        position: fixed;
        right: 10px;
        top: 6.05rem;
        width: 270px;
        height: calc(100vh - 6.75rem);
        overflow-y: auto;
        padding: 0.65rem 0.65rem 8.2rem 0.65rem;
        box-sizing: border-box;
        border: 1px solid rgba(124, 58, 237, 0.42);
        border-radius: 0.75rem;
        background: rgba(15, 23, 42, 0.48);
        z-index: 998;
    }
    .st-key-chat_input_bar {
        position: fixed;
        right: 18px;
        bottom: 10px;
        width: 254px;
        z-index: 1002;
        padding: 0.55rem 0.55rem 0.35rem;
        box-sizing: border-box;
        border-top: 1px solid rgba(124, 58, 237, 0.35);
        border-radius: 0.65rem;
        background: rgba(15, 23, 42, 0.96);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.28);
    }
    .st-key-chat_input_bar textarea {
        min-height: 4.25rem !important;
        max-height: 4.25rem !important;
    }
    .st-key-canvas_panel {
        min-height: calc(100vh - 6rem);
    }
    .agent-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }
    .nav-title {
        font-weight: 700;
        font-size: 0.84rem;
        margin-bottom: 0.6rem;
        color: #c4b5fd;
    }
    .st-key-nav_panel label,
    .st-key-nav_panel p,
    .st-key-nav_panel span {
        font-size: 0.82rem !important;
    }

    @media (max-width: 1050px) {
        div.block-container {
            padding-left: 150px;
            padding-right: 280px;
        }
        .st-key-nav_panel { width: 128px; }
        .st-key-agent_panel { width: 260px; }
        .st-key-chat_input_bar { width: 244px; }
    }
</style>
    """,
    unsafe_allow_html=True,
)


from gui.panels import chat as chat_panel  # noqa: E402
from gui.panels import (  # noqa: E402
    eda as eda_panel,
    entregables as entregables_panel,
    experiments as exp_panel,
    explorer as data_panel,
    proposals as prop_panel,
    status as status_panel,
)


# ---------- render genérico de artifacts ----------------------------------


def _coerce_df(rows) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows
    if isinstance(rows, list):
        try:
            return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def _render_metrics(items: list[dict]) -> None:
    if not items:
        return
    cols = st.columns(min(len(items), 4))
    for i, it in enumerate(items[:4]):
        col = cols[i % len(cols)]
        label = str(it.get("label", "—"))
        value = it.get("value", "")
        if isinstance(value, float):
            value = f"{value:,.2f}"
        elif isinstance(value, int):
            value = f"{value:,}"
        else:
            value = str(value)
        col.metric(label, value, help=it.get("help"))
    rest = items[4:]
    if rest:
        cols2 = st.columns(min(len(rest), 4))
        for i, it in enumerate(rest):
            col = cols2[i % len(cols2)]
            value = it.get("value", "")
            if isinstance(value, float):
                value = f"{value:,.2f}"
            elif isinstance(value, int):
                value = f"{value:,}"
            else:
                value = str(value)
            col.metric(str(it.get("label", "—")), value, help=it.get("help"))


def _render_chart(art: dict) -> None:
    fig = art.get("figure")
    if fig is not None:
        try:
            st.plotly_chart(fig, width="stretch")
            return
        except Exception as e:  # noqa: BLE001
            st.caption(f"No pude renderizar la figura: {e}")
    df = _coerce_df(art.get("data") or [])
    if df.empty:
        st.caption("Sin datos para graficar.")
        return
    chart_type = (art.get("chart_type") or "bar").lower()
    x = art.get("x") or (df.columns[0] if len(df.columns) else None)
    y = art.get("y") or (df.columns[1] if len(df.columns) > 1 else None)
    try:
        if chart_type == "line" and x and y:
            st.line_chart(df.set_index(x)[y] if y in df.columns else df, height=320)
        elif chart_type == "scatter" and x and y:
            st.scatter_chart(df, x=x, y=y, height=320)
        elif chart_type == "histogram" and x:
            st.bar_chart(df[x].value_counts().head(30), height=320)
        else:  # bar default
            if x and y and y in df.columns:
                st.bar_chart(df.set_index(x)[y], height=320)
            else:
                st.bar_chart(df, height=320)
    except Exception as e:  # noqa: BLE001
        st.caption(f"No pude graficar ({e}). Te dejo la tabla:")
        st.dataframe(df, width="stretch", hide_index=True)


def _render_one_artifact(art: dict) -> None:
    kind = (art.get("kind") or "table").lower()
    title = art.get("title") or "Resultado"
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if kind == "metrics":
            _render_metrics(art.get("items") or [])
            if art.get("caption"):
                st.caption(art["caption"])
        elif kind == "chart":
            _render_chart(art)
            if art.get("caption"):
                st.caption(art["caption"])
            if art.get("summary"):
                st.caption(str(art["summary"])[:600])
        elif kind == "markdown":
            st.markdown(art.get("text") or "")
        else:  # table
            df = _coerce_df(art.get("data") or [])
            if df.empty:
                st.caption("Sin filas para mostrar.")
            else:
                st.dataframe(df, width="stretch", hide_index=True)
                shape = art.get("shape")
                cap = art.get("caption")
                if shape:
                    st.caption(f"{shape[0]} filas × {shape[1]} columnas"
                               + (f" — {cap}" if cap else ""))
                elif cap:
                    st.caption(cap)


def render_agent_artifacts() -> None:
    """Renderiza la lista de artifacts del último turno del agente."""
    artifacts = st.session_state.get("agent_artifacts") or []
    if not artifacts:
        return

    with st.container(border=False):
        head_l, head_r = st.columns([0.85, 0.15], vertical_alignment="center")
        with head_l:
            st.markdown(
                f"### Resultados del agente "
                f"<span style='opacity:0.6;font-size:0.85rem'>({len(artifacts)} item{'s' if len(artifacts) != 1 else ''})</span>",
                unsafe_allow_html=True,
            )
        with head_r:
            if st.button("Limpiar", key="clear_agent_artifacts", width="stretch"):
                st.session_state.agent_artifacts = []
                st.rerun()

        for i, art in enumerate(artifacts):
            try:
                _render_one_artifact(art)
            except Exception as e:  # noqa: BLE001
                st.error(f"No pude renderizar el artifact #{i}: {e}")


def main() -> None:
    # === Header FIJO ===
    st.markdown(
        """
<div class="app-fixed-header">
  <h1>📊 Chatea con los datos</h1>
  <div class="subtitle">Competencia Kaggle · viviendas CABA — Maestría en Explotación de Datos, FCEN-UBA</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="nav_panel"):
        st.markdown('<div class="nav-title">Navegación</div>', unsafe_allow_html=True)
        vista = st.radio(
            "Qué querés hacer",
            [
                "Inicio",
                "Explorar datos",
                "Experimentos",
                "Entregables",
                "Análisis EDA",
                "Cambios sugeridos",
            ],
            horizontal=False,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Canvas central + chat fijo a la derecha.")

    with st.container(key="agent_panel"):
        st.markdown('<div class="agent-title">🤖 Asistente de datos</div>',
                    unsafe_allow_html=True)
        st.caption("Panel fijo. Los resultados visuales aparecen en el canvas central.")
        chat_panel.render()

    with st.container(key="canvas_panel"):
        render_agent_artifacts()

        if vista == "Inicio":
            status_panel.render()
        elif vista == "Explorar datos":
            with st.spinner("Cargando vista de datos..."):
                data_panel.render()
        elif vista == "Experimentos":
            exp_panel.render()
        elif vista == "Entregables":
            entregables_panel.render()
        elif vista == "Análisis EDA":
            with st.spinner("Cargando EDA interactivo..."):
                eda_panel.render()
        elif vista == "Cambios sugeridos":
            prop_panel.render()


if __name__ == "__main__":
    main()
