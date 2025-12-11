#!/usr/bin/env python3
"""
Interactive Streamlit app for exploring a prepped scRNA-seq AnnData (.h5ad) file.

Run locally:
    streamlit run scrna_viewer_app.py -- --data-path /path/to/prepped_object.h5ad
You can also use the sidebar to upload a .h5ad if you prefer not to pass a path.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import anndata as ad
    from scipy import sparse
except ImportError:
    ad = None
    sparse = None


def parse_cli_args() -> argparse.Namespace:
    """Capture optional CLI arguments passed after `--` in `streamlit run`."""
    parser = argparse.ArgumentParser(description="Interactive viewer for prepped scRNA-seq AnnData files.")
    parser.add_argument("--data-path", type=str, default="", help="Path to the prepped .h5ad file.")
    parser.add_argument("--title", type=str, default="scRNA-seq .h5ad Explorer", help="Title shown at the top of the page.")
    args, _ = parser.parse_known_args()
    return args


@st.cache_resource(show_spinner="Reading .h5ad file...")
def load_anndata(path_str: str):
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Could not find .h5ad at {path}")
    return ad.read_h5ad(path)


def find_embedding_keys(adata) -> List[str]:
    """Return obsm keys that look like 2D embeddings."""
    preferred = ["X_umap", "X_tsne", "X_fumap", "X_draw_graph_fa", "X_pca"]
    keys: List[str] = []
    for key, value in adata.obsm.items():
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            keys.append(key)
    sorted_keys = sorted(keys, key=lambda k: preferred.index(k) if k in preferred else len(preferred))
    return sorted_keys


def get_categorical_obs_columns(adata, max_unique: int = 50) -> List[str]:
    """Columns suitable for grouping/filtering (categoricals or few unique values)."""
    cols: List[str] = []
    for col in adata.obs.columns:
        series = adata.obs[col]
        if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
            if series.nunique(dropna=False) <= max_unique:
                cols.append(col)
        elif series.nunique(dropna=False) <= max_unique:
            cols.append(col)
    return cols


def downsample_indices(n: int, max_points: int) -> Iterable[int]:
    if n <= max_points:
        return slice(None)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def stratified_downsample(
    adata, group_col: str, max_points: int, seed: int = 0, min_per_group: int = 1
):
    """Downsample while keeping representation from each group in obs[group_col]."""
    n = adata.n_obs
    if n <= max_points:
        return adata
    groups = adata.obs[group_col].astype(str)
    unique_groups = groups.unique().tolist()
    rng = np.random.default_rng(seed)
    per_group = max(min_per_group, max_points // max(len(unique_groups), 1))
    keep_indices: List[int] = []
    for g in unique_groups:
        g_idx = np.where(groups == g)[0]
        take = min(len(g_idx), per_group)
        if take > 0:
            keep_indices.extend(rng.choice(g_idx, size=take, replace=False).tolist())
    if len(keep_indices) < max_points:
        remaining = np.setdiff1d(np.arange(n), keep_indices, assume_unique=False)
        fill = min(max_points - len(keep_indices), len(remaining))
        if fill > 0:
            keep_indices.extend(rng.choice(remaining, size=fill, replace=False).tolist())
    return adata[np.sort(keep_indices)]


def categorical_legend_chart(categories: List[str], scheme: str, title: str):
    """Render a simple categorical legend as a stand-alone Altair chart."""
    df = pd.DataFrame({"category": categories})
    return (
        alt.Chart(df)
        .mark_rect(height=14, width=14)
        .encode(
            y=alt.Y("category:N", sort=categories, title=title),
            color=alt.Color("category:N", scale=alt.Scale(scheme=scheme), legend=None),
        )
        .properties(width=200, height=max(80, 16 * len(categories)))
    ) + (
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", dx=18)
        .encode(y=alt.Y("category:N", sort=categories, title=None), text="category")
        .properties(width=200, height=max(80, 16 * len(categories)))
    )


def continuous_legend_chart(scheme: str, title: str):
    """Render a simple continuous color bar legend."""
    df = pd.DataFrame({"value": np.linspace(0, 1, 30)})
    return (
        alt.Chart(df)
        .mark_rect(height=20)
        .encode(
            x=alt.X("value:Q", title=title),
            color=alt.Color("value:Q", scale=alt.Scale(scheme=scheme), legend=None),
        )
        .properties(width=220, height=40)
    )


def extract_gene_vector(adata, gene: str, layer_choice: Optional[str], use_raw: bool) -> np.ndarray:
    """Pull gene expression as a dense vector from the requested matrix."""
    matrix = adata
    var_names = adata.var_names

    if use_raw and adata.raw is not None:
        matrix = adata.raw
        var_names = adata.raw.var_names
    elif layer_choice and layer_choice != "X":
        matrix_data = adata.layers[layer_choice]
    else:
        matrix_data = adata.X

    if use_raw and adata.raw is not None:
        matrix_data = matrix.X

    if gene not in var_names:
        raise KeyError(f"Gene '{gene}' not found in {('raw' if use_raw else 'var')} names.")

    idx = var_names.get_loc(gene)
    values = matrix_data[:, idx]

    if sparse is not None and sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def build_embedding_df(adata, embedding_key: str, color_by: Optional[str]) -> pd.DataFrame:
    arr = np.asarray(adata.obsm[embedding_key])
    df = pd.DataFrame(arr[:, :2], columns=[f"{embedding_key}_1", f"{embedding_key}_2"])
    if color_by:
        df["color"] = adata.obs[color_by].values
    return df


def main() -> None:
    args = st.session_state.get("cli_args")
    if args is None:
        args = parse_cli_args()
        st.session_state["cli_args"] = args

    st.set_page_config(page_title=args.title, layout="wide")
    st.title(args.title)
    st.caption(
        "Load a prepped AnnData (.h5ad) file, then explore embeddings, metadata and gene expression "
        "without leaving your browser."
    )

    if ad is None:
        st.error("Install dependencies first: pip install streamlit anndata scipy pandas altair")
        return

    st.sidebar.header("Data input")
    data_path = st.sidebar.text_input("Path to .h5ad", args.data_path)
    uploaded = st.sidebar.file_uploader("Or upload a .h5ad file", type=["h5ad"])
    if uploaded:
        tmp_path = Path(tempfile.gettempdir()) / uploaded.name
        tmp_path.write_bytes(uploaded.read())
        data_path = str(tmp_path)
        st.sidebar.info(f"Using uploaded file: {tmp_path}")

    if not data_path:
        st.info("Provide a path or upload a prepped .h5ad to begin.")
        return

    try:
        adata = load_anndata(data_path)
    except Exception as exc:
        st.error(f"Could not load AnnData: {exc}")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Cells", f"{adata.n_obs:,}")
    col_b.metric("Genes", f"{adata.n_vars:,}")
    col_c.metric("Embeddings", f"{len(adata.obsm_keys())}")
    col_d.metric("Layers", f"{len(adata.layers.keys())}")

    embedding_options = find_embedding_keys(adata)
    if not embedding_options:
        st.warning("No 2D embeddings were found in .obsm (e.g., X_umap, X_tsne).")
        return

    categorical_cols = get_categorical_obs_columns(adata)
    numeric_obs_cols = [c for c in adata.obs.columns if pd.api.types.is_numeric_dtype(adata.obs[c])]

    st.sidebar.header("Filtering")
    filter_col = st.sidebar.selectbox("Filter by category", options=["(none)"] + categorical_cols)
    selected_categories: List[str] = []
    filtered_adata = adata
    if filter_col and filter_col != "(none)":
        categories = sorted(adata.obs[filter_col].astype(str).unique().tolist())
        selected_categories = st.sidebar.multiselect("Keep categories", options=categories, default=categories)
        mask = adata.obs[filter_col].astype(str).isin(selected_categories)
        filtered_adata = adata[mask].copy()

    st.sidebar.header("Plotting")
    embedding_choice = st.sidebar.selectbox("Embedding", options=embedding_options, index=0)
    max_cells = st.sidebar.slider("Max cells to draw", min_value=500, max_value=50000, value=15000, step=500)
    color_by = st.sidebar.selectbox(
        "Color by (metadata)", options=["(none)"] + categorical_cols + numeric_obs_cols
    )
    categorical_palette = st.sidebar.selectbox(
        "Color palette (categorical)", options=["tableau10", "category10", "set1", "set2", "dark2", "paired"], index=0
    )
    continuous_palette = st.sidebar.selectbox(
        "Color palette (continuous)", options=["viridis", "plasma", "magma", "inferno", "turbo", "redblue"], index=0
    )

    plot_indices = downsample_indices(filtered_adata.n_obs, max_cells)
    plot_adata = filtered_adata[plot_indices].copy()
    plot_df = build_embedding_df(plot_adata, embedding_choice, None if color_by == "(none)" else color_by)

    st.subheader("Embedding view")
    color_field = "color" if color_by != "(none)" else None
    tooltip_fields = list(plot_df.columns[:2])
    if color_field:
        tooltip_fields.append(color_field)

    color_encoding = None
    if color_field:
        is_numeric = pd.api.types.is_numeric_dtype(plot_adata.obs[color_by])
        scheme = continuous_palette if is_numeric else categorical_palette
        legend = alt.Legend(title=color_by)
        color_encoding = alt.Color(color_field, title=color_by, scale=alt.Scale(scheme=scheme), legend=legend)

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=25, opacity=0.65)
        .encode(
            x=plot_df.columns[0],
            y=plot_df.columns[1],
            color=color_encoding if color_field else alt.value("#1f77b4"),
            tooltip=tooltip_fields,
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Facet panels by category")
    facet_col = st.selectbox(
        "Facet by (categorical obs)", options=["(none)"] + categorical_cols, help="Creates small-multiple UMAPs."
    )
    facet_color_by = st.selectbox(
        "Color by (facet view)", options=["(none)"] + categorical_cols + numeric_obs_cols, index=0
    )
    max_facets = st.slider("Max facets to display", min_value=2, max_value=24, value=12, step=1)
    facet_columns = st.slider("Facet columns", min_value=1, max_value=6, value=3, step=1)

    if facet_col and facet_col != "(none)":
        # Determine which facet categories to show (up to max_facets).
        facet_series = filtered_adata.obs[facet_col].astype(str)
        top_facets = facet_series.value_counts().index.tolist()[:max_facets]
        facet_mask = facet_series.isin(top_facets)
        facet_source = filtered_adata[facet_mask].copy()

        # Downsample with representation from each facet category to keep all top facets visible.
        facet_adata = stratified_downsample(facet_source, facet_col, max_cells)
        facet_df = build_embedding_df(facet_adata, embedding_choice, None)
        facet_df["facet"] = facet_adata.obs[facet_col].astype(str).reset_index(drop=True).values

        facet_color_field = None if facet_color_by == "(none)" else "facet_color"
        if facet_color_field:
            color_values = facet_adata.obs[facet_color_by].reset_index(drop=True)
            facet_df[facet_color_field] = color_values
        is_numeric = facet_color_field and pd.api.types.is_numeric_dtype(facet_df[facet_color_field])
        scheme = continuous_palette if is_numeric else categorical_palette
        color_type = "quantitative" if is_numeric else "nominal"
        color_enc = (
            alt.Color(
                facet_color_field,
                type=color_type,
                title=facet_color_by,
                scale=alt.Scale(scheme=scheme),
                legend=None,  # hide per-panel legend; show master legend below controls
            )
            if facet_color_field
            else alt.value("#1f77b4")
        )

        facet_tooltips = [c for c in facet_df.columns if c not in ["facet_color"]]
        if facet_color_field:
            facet_tooltips.append(facet_color_field)

        facet_chart = (
            alt.Chart(facet_df)
            .mark_circle(size=20, opacity=0.7)
            .encode(
                x=plot_df.columns[0],
                y=plot_df.columns[1],
                color=color_enc,
                tooltip=facet_tooltips,
                facet=alt.Facet("facet:N", columns=facet_columns, sort=top_facets),
            )
            .properties(width=200, height=200)
        )
        st.altair_chart(facet_chart, use_container_width=True)

        # Master legend under the facet controls.
        if facet_color_field:
            if is_numeric:
                legend_chart = continuous_legend_chart(scheme=scheme, title=facet_color_by)
            else:
                categories = sorted(pd.unique(facet_df[facet_color_field]))
                legend_chart = categorical_legend_chart(categories=categories, scheme=scheme, title=facet_color_by)
            st.altair_chart(legend_chart, use_container_width=False)
    else:
        st.caption("Select a faceting category to view a panel of UMAPs.")

    st.subheader("Gene expression on embedding")
    layer_options = ["X"] + sorted(adata.layers.keys())
    layer_choice = st.selectbox("Expression matrix / layer", options=layer_options, index=0)
    use_raw = st.checkbox("Use .raw (if available)", value=adata.raw is not None)

    gene_search = st.text_input("Find gene (substring match against var_names)", value="")
    max_matches = 500
    if gene_search:
        all_matches = [g for g in adata.var_names if gene_search.lower() in g.lower()]
    else:
        all_matches = list(adata.var_names)
    truncated = len(all_matches) > max_matches
    matching_genes = all_matches[:max_matches]
    if truncated:
        st.caption(f"Showing first {max_matches} matches out of {len(all_matches)} total.")
    selected_gene = st.selectbox("Gene to plot", options=matching_genes or ["(no matches)"])

    # If the exact search term matches a gene not surfaced (e.g., beyond max_matches), use it.
    if selected_gene == "(no matches)" and gene_search:
        exact = [g for g in adata.var_names if g.lower() == gene_search.lower()]
        if exact:
            selected_gene = exact[0]

    if selected_gene and selected_gene != "(no matches)":
        try:
            expr = extract_gene_vector(plot_adata, selected_gene, layer_choice, use_raw)
            expr_df = plot_df.copy()
            expr_df["expression"] = expr

            expr_palette = st.selectbox(
                "Gene expression palette", options=["viridis", "plasma", "magma", "inferno", "turbo", "redblue"], index=0
            )

            expr_chart = (
                alt.Chart(expr_df)
                .mark_circle(size=25, opacity=0.7)
                .encode(
                    x=plot_df.columns[0],
                    y=plot_df.columns[1],
                    color=alt.Color("expression", scale=alt.Scale(scheme=expr_palette)),
                    tooltip=tooltip_fields + ["expression"],
                )
                .interactive()
            )
            st.altair_chart(expr_chart, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not plot expression for {selected_gene}: {exc}")

    st.subheader("Metadata preview")
    st.write("First few rows from `obs`:")
    st.dataframe(plot_adata.obs.head(15))

    st.caption(
        "Tip: rerun with `streamlit run scrna_viewer_app.py -- --data-path /path/to/file.h5ad` "
        "to load a specific object without using the uploader."
    )


if __name__ == "__main__":
    main()
