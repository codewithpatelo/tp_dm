"""Panel Data Explorer: filtros + stats live + plots + map."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..data_loader import DataRegistry


def render():
    st.subheader("Explorar datos")
    reg = DataRegistry.get()
    table = st.selectbox(
        "Vista (tabla)",
        ["train_filtered", "test"],
        index=0,
        help="Entrenamiento filtrado (E1) o conjunto a predecir.",
    )
    df = reg.table(table)

    with st.expander("Filtros", expanded=True):
        cols = st.columns(3)
        barrios = sorted([b for b in df.get("location_2", pd.Series([])).dropna().unique()])[:80]
        sel_barrios = cols[0].multiselect("Barrio (location_2)", barrios)
        prop_types = sorted(df.get("property_type", pd.Series([])).dropna().unique())
        sel_props = cols[1].multiselect("Tipo de propiedad", prop_types,
                                        default=list(prop_types))
        if "price" in df.columns:
            pmin, pmax = float(df["price"].min()), float(df["price"].max())
            sel_price = cols[2].slider("Precio USD",
                                       min_value=pmin, max_value=pmax,
                                       value=(pmin, pmax))
        else:
            sel_price = None

    sub = df.copy()
    if sel_barrios:
        sub = sub[sub["location_2"].isin(sel_barrios)]
    if sel_props:
        sub = sub[sub["property_type"].isin(sel_props)]
    if sel_price is not None and "price" in sub.columns:
        sub = sub[sub["price"].between(sel_price[0], sel_price[1])]

    st.caption(f"{len(sub):,} filas · {len(sub.columns)} columnas")

    cstats = st.columns(4)
    cstats[0].metric("Filas", f"{len(sub):,}")
    if "price" in sub.columns and len(sub):
        cstats[1].metric("Precio mediana", f"${sub['price'].median():,.0f}")
        cstats[2].metric("Precio media", f"${sub['price'].mean():,.0f}")
    if "surface_total_in_m2" in sub.columns and sub["surface_total_in_m2"].notna().any():
        cstats[3].metric("Sup. mediana m²", f"{sub['surface_total_in_m2'].median():,.0f}")

    tabs = st.tabs(
        ["Tabla", "Histograma de precio", "Sup. vs precio", "Mapa (lat/lon)"]
    )
    with tabs[0]:
        st.dataframe(sub.head(500), use_container_width=True)
    with tabs[1]:
        if "price" in sub.columns and len(sub):
            fig = px.histogram(sub, x="price", nbins=50, title="Distribución del precio (USD)")
            st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        if {"surface_total_in_m2", "price"}.issubset(sub.columns) and len(sub):
            sample = sub.sample(min(5000, len(sub)), random_state=42)
            fig = px.scatter(sample, x="surface_total_in_m2", y="price",
                             color="property_type", opacity=0.5,
                             title="Sup total vs precio")
            st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        if {"lat", "lon"}.issubset(sub.columns):
            geo = sub[["lat", "lon"]].dropna()
            geo = geo[(geo["lat"].between(-35, -34)) & (geo["lon"].between(-59, -58))]
            geo = geo.sample(min(3000, len(geo)), random_state=42)
            st.map(geo, latitude="lat", longitude="lon")
        else:
            st.caption("No hay lat/lon en esta vista.")
