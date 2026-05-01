"""Vista del informe (Markdown / PDF) y del notebook campeón por entrega."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ..parsers import EntregaEntregables, list_entregas_con_entregables, ROOT


def _render_notebook_preview(path: Path, max_cells: int = 120) -> None:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    st.download_button(
        "Descargar `.ipynb`",
        data=raw,
        file_name=path.name,
        mime="application/json",
        type="primary",
    )
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"No pude leer el notebook como JSON: {e}")
        return
    cells = nb.get("cells", [])
    st.caption(
        f"{len(cells)} celdas — vista previa de las primeras {min(max_cells, len(cells))} "
        "(el resto está en el archivo descargable)."
    )
    for i, cell in enumerate(cells[:max_cells]):
        src = cell.get("source", [])
        if isinstance(src, list):
            text = "".join(src)
        else:
            text = str(src)
        if not text.strip():
            continue
        ct = cell.get("cell_type", "code")
        with st.expander(f"Celda {i + 1} · {ct}", expanded=(i < 2 and len(text) < 2000)):
            if ct == "markdown":
                st.markdown(text)
            else:
                st.code(text, language="python" if ct == "code" else "text")


def render() -> None:
    st.caption(
        "Informe en Markdown o PDF y `notebook_campeon.ipynb` que guardes en "
        "`entregas/entrega_n/` (convención del TP)."
    )
    st.subheader("Informe y notebook campeón")

    entregas = list_entregas_con_entregables()
    if not entregas:
        st.warning(
            "No hay entregables detectados. Colocá en `entregas/entrega_N/` al menos un archivo "
            "`*informe*.md` y/o `notebook_campeon.ipynb`."
        )
        return

    # Orden: entrega_1, entrega_2, …
    def _sort_key(e: EntregaEntregables) -> tuple:
        s = e.carpeta.replace("entrega_", "")
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    entregas = sorted(entregas, key=_sort_key)
    if len(entregas) > 1:
        e: EntregaEntregables = st.selectbox(
            "Elegir entrega",
            entregas,
            format_func=lambda x: x.carpeta,
        )
    else:
        e = entregas[0]

    st.markdown(f"**Carpeta:** `{e.ruta.relative_to(ROOT)}`")

    sub = st.tabs(["Informe", "Notebook campeón"])
    with sub[0]:
        if e.informe_pdf:
            for pdf in e.informe_pdf:
                data = Path(pdf).read_bytes()
                st.download_button(
                    f"Descargar PDF · {pdf.name}",
                    data=data,
                    file_name=pdf.name,
                    mime="application/pdf",
                )
        if e.informe_md and e.informe_md.exists():
            md = e.informe_md.read_text(encoding="utf-8", errors="ignore")
            st.download_button(
                "Descargar `.md`",
                data=md,
                file_name=e.informe_md.name,
                mime="text/markdown",
            )
            st.divider()
            st.markdown(md)
        elif not e.informe_pdf:
            st.info("No hay `*informe*.md` ni PDF en esta carpeta.")

    with sub[1]:
        if e.notebook_campeon and e.notebook_campeon.exists():
            st.caption(f"Archivo: `{e.notebook_campeon.name}`")
            _render_notebook_preview(e.notebook_campeon)
        else:
            st.info(
                "No se encontró `notebook_campeon.ipynb` (u otro `notebook_campeon*.ipynb`) en esta entrega."
            )
