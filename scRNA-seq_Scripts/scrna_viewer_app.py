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
import gseapy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scanpy as sc
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


@st.cache_data(show_spinner="Running DGE analysis...")
def run_dge(_adata: ad.AnnData, groupby: str, group1: str, group2: str, method: str) -> pd.DataFrame:
    """Run scanpy's rank_genes_groups and return a clean DataFrame."""
    # rank_genes_groups modifies the adata object, so work on a copy
    adata_copy = _adata.copy()
    sc.tl.rank_genes_groups(
        adata_copy,
        groupby=groupby,
        groups=[group1],
        reference=group2,
        method=method,
    )
    # Extract results into a tidy dataframe
    result_df = sc.get.rank_genes_groups_df(adata_copy, group=group1)
    return result_df


@st.cache_data(show_spinner="Running GSEA...")
def run_gsea(gene_list: List[str], gene_sets: str, organism: str = "Human") -> pd.DataFrame:
    """Run GSEA using gseapy.enrichr."""
    enr = gseapy.enrichr(
        gene_list=gene_list,
        gene_sets=[gene_sets],
        organism=organism,
    )
    return enr.results


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


def categorical_legend_chart(categories: List[str], scheme: str, title: str, orient: str = "vertical"):
    """Render a simple categorical legend as a stand-alone Altair chart."""
    df = pd.DataFrame({"category": categories})
    if not categories:
        return alt.Chart(pd.DataFrame()).mark_text().properties(height=10)

    if orient == "horizontal":
        items_per_row = 8 
        df_h = pd.DataFrame({
            "category": categories,
            "row": np.arange(len(categories)) // items_per_row,
            "col": np.arange(len(categories)) % items_per_row
        })
        base = alt.Chart(df_h)
        circles = base.mark_circle(size=100).encode(
            y=alt.Y('row:O', axis=None, title=title),
            x=alt.X('col:O', axis=None),
            color=alt.Color("category:N", scale=alt.Scale(scheme=scheme, domain=categories), legend=None)
        )
        texts = base.mark_text(align="left", dx=12, baseline="middle").encode(
            y=alt.Y('row:O', axis=None),
            x=alt.X('col:O', axis=None),
            text="category:N"
        )
        chart = alt.layer(circles, texts).configure_view(strokeWidth=0)
        num_rows = (df_h['row'].max() + 1) if not df_h.empty else 0
        return chart.properties(height=num_rows * 25)

    # Default to vertical
    base = alt.Chart(df).properties(height=max(80, 20 * len(categories)))
    circles = base.mark_circle(size=80).encode(
        y=alt.Y("category:N", sort=categories, title=title, axis=alt.Axis(labelAngle=0, titlePadding=10, grid=False)),
        color=alt.Color("category:N", scale=alt.Scale(scheme=scheme, domain=categories), legend=None),
    )
    texts = base.mark_text(align="left", baseline="middle", dx=10).encode(
        y=alt.Y("category:N", sort=categories, title=None),
        text="category:N"
    )
    return alt.layer(circles, texts).configure_view(strokeWidth=0)


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


def format_long_string(s: str, max_len: int = 35) -> str:
    """Truncate long strings for display in UI elements."""
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s


def main() -> None:
    args = st.session_state.get("cli_args")
    if args is None:
        args = parse_cli_args()
        st.session_state["cli_args"] = args
    
    # Initialize states
    if 'zoom_reset_key' not in st.session_state:
        st.session_state.zoom_reset_key = 0
    if "filters" not in st.session_state:
        st.session_state.filters = {}

    st.set_page_config(page_title=args.title, layout="wide")
    st.title(args.title)
    st.caption(
        "Load a prepped AnnData (.h5ad) file, then explore embeddings, metadata and gene expression "
        "without leaving your browser."
    )

    if ad is None:
        st.error("Install dependencies first: pip install streamlit anndata scipy pandas altair plotly")
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

    st.sidebar.header("Filtering")
    
    # Apply existing filters from session state
    filtered_adata = adata
    if st.session_state.filters:
        for column, selected_values in st.session_state.filters.items():
            mask = filtered_adata.obs[column].astype(str).isin(selected_values)
            filtered_adata = filtered_adata[mask].copy()
    
    # --- Display metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cells", f"{adata.n_obs:,}")
    col2.metric("Filtered Cells", f"{filtered_adata.n_obs:,}")
    col3.metric("Genes", f"{adata.n_vars:,}")
    col4.metric("Embeddings", f"{len(adata.obsm_keys())}")


    embedding_options = find_embedding_keys(adata)
    if not embedding_options:
        st.warning("No 2D embeddings were found in .obsm (e.g., X_umap, X_tsne).")
        return

    categorical_cols = get_categorical_obs_columns(adata)
    numeric_obs_cols = [c for c in adata.obs.columns if pd.api.types.is_numeric_dtype(adata.obs[c])]

    # --- NEW FILTERING UI ---
    # UI to add a new filter
    add_filter_col = st.sidebar.selectbox(
        "Add filter on column",
        options=["(none)"] + [c for c in categorical_cols if c not in st.session_state.filters],
        key="add_filter_select"
    )
    if add_filter_col != "(none)":
        all_values = sorted(adata.obs[add_filter_col].astype(str).unique().tolist())
        st.session_state.filters[add_filter_col] = all_values
        st.rerun()

    # Display active filters
    if st.session_state.filters:
        st.sidebar.write("Active filters:")
        for column, selected_values in list(st.session_state.filters.items()):
            with st.sidebar.expander(f"Filter by: {column}", expanded=True):
                all_values = sorted(adata.obs[column].astype(str).unique().tolist())
                
                new_selections = st.multiselect(
                    "Keep values:",
                    options=all_values,
                    default=selected_values,
                    key=f"ms_{column}",
                    format_func=format_long_string,
                )

                # Check for changes and update state
                if set(new_selections) != set(selected_values):
                    st.session_state.filters[column] = new_selections
                    st.rerun()

                if st.button("Remove Filter", key=f"remove_{column}", use_container_width=True):
                    del st.session_state.filters[column]
                    st.rerun()
        
        if st.sidebar.button("Clear All Filters", use_container_width=True):
            st.session_state.filters = {}
            st.rerun()

    st.sidebar.header("Plotting")
    embedding_choice = st.sidebar.selectbox("Embedding", options=embedding_options, index=0)
    max_cells = st.sidebar.slider("Max cells to draw", min_value=500, max_value=50000, value=15000, step=500)
    color_by = st.sidebar.selectbox(
        "Color by (metadata)", options=["(none)"] + categorical_cols + numeric_obs_cols
    )
    categorical_palette = st.sidebar.selectbox(
        "Color palette (categorical)", options=["tableau10", "category10", "set1", "set2", "dark2"], index=0
    )
    continuous_palette = st.sidebar.selectbox(
        "Color palette (continuous)", options=["viridis", "plasma", "magma", "inferno", "turbo", "redblue"], index=0
    )

    # Map Altair/Vega scheme names to Plotly's qualitative color lists
    VEGA_TO_PLOTLY_CAT = {
        "tableau10": px.colors.qualitative.T10,
        "category10": px.colors.qualitative.G10,
        "set1": px.colors.qualitative.Set1,
        "set2": px.colors.qualitative.Set2,
        "dark2": px.colors.qualitative.Dark2,
    }
    plotly_palette = VEGA_TO_PLOTLY_CAT.get(categorical_palette, px.colors.qualitative.T10)
    plot_indices = downsample_indices(filtered_adata.n_obs, max_cells)
    plot_adata = filtered_adata[plot_indices].copy()
    plot_df = build_embedding_df(plot_adata, embedding_choice, None if color_by == "(none)" else color_by)

    tab1, tab2, tab3 = st.tabs(["Embedding Visualizations", "Composition Analysis", "Differential Expression"])

    with tab1:
        st.subheader("Embedding view")
        color_field = "color" if color_by != "(none)" else None

        # Create Plotly figure
        fig = go.Figure()

        # Determine color properties
        color_data = None
        if color_field:
            is_numeric = pd.api.types.is_numeric_dtype(plot_adata.obs[color_by])
            colors = plot_adata.obs[color_by]
            if is_numeric:
                color_data = colors
                fig.add_trace(go.Scattergl(
                    x=plot_df.iloc[:, 0],
                    y=plot_df.iloc[:, 1],
                    mode='markers',
                    marker=dict(
                        color=color_data,
                        colorscale=continuous_palette,
                        showscale=True,
                        colorbar=dict(title=color_by)
                    ),
                    text=plot_adata.obs[color_by],
                    hoverinfo='text'
                ))
            else:
                # For categorical, we need to map categories to colors
                unique_cats = colors.unique()
                palette = plotly_palette
                cat_to_color = {cat: color for cat, color in zip(unique_cats, palette)}
                color_data = colors.map(cat_to_color)
                
                # Add a trace for each category to build a legend
                for cat in unique_cats:
                    cat_mask = (colors == cat).values  # Use .values to ignore index
                    fig.add_trace(go.Scattergl(
                        x=plot_df.iloc[:, 0][cat_mask],
                        y=plot_df.iloc[:, 1][cat_mask],
                        mode='markers',
                        marker=dict(color=cat_to_color.get(cat)), # Use .get for safety
                        name=cat,
                        text=colors[cat_mask],
                        hoverinfo='text'
                    ))

        else:
            # No color
            fig.add_trace(go.Scattergl(
                x=plot_df.iloc[:, 0],
                y=plot_df.iloc[:, 1],
                mode='markers',
                marker=dict(color="#1f77b4")
            ))

        fig.update_layout(
            width=1000,
            height=1000,
            xaxis_title=plot_df.columns[0],
            yaxis_title=plot_df.columns[1],
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            showlegend=(not (color_field and is_numeric)), # Show legend for categorical
            template="simple_white"
        )
        st.plotly_chart(fig)

        st.subheader("Facet panels by category")
        facet_col = st.selectbox(
            "Facet by (categorical obs)", options=["(none)"] + categorical_cols, help="Creates small-multiple UMAPs."
        )
        facet_color_by = st.selectbox(
            "Color by (facet view)", options=["(none)"] + categorical_cols + numeric_obs_cols, index=0
        )
        max_facets = st.slider("Max facets to display", min_value=2, max_value=24, value=12, step=1)
        facet_columns = st.slider("Facet columns", min_value=1, max_value=6, value=4, step=1)

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

            # Define shared domains and encodings
            x_col, y_col = plot_df.columns[0], plot_df.columns[1]
            x_domain = (facet_df[x_col].min(), facet_df[x_col].max())
            y_domain = (facet_df[y_col].min(), facet_df[y_col].max())

            is_numeric = facet_color_field and pd.api.types.is_numeric_dtype(facet_df[facet_color_field])
            scheme = continuous_palette if is_numeric else categorical_palette
            color_type = "quantitative" if is_numeric else "nominal"

            color_scale = alt.Scale(scheme=scheme)
            if is_numeric and facet_color_field in facet_df:
                color_domain = (facet_df[facet_color_field].min(), facet_df[facet_color_field].max())
                color_scale.domain = list(color_domain)

            color_enc = (
                alt.Color(
                    facet_color_field,
                    type=color_type,
                    title=facet_color_by,
                    scale=color_scale,
                    legend=None,
                )
                if facet_color_field
                else alt.value("#1f77b4")
            )

            facet_tooltips = [c for c in facet_df.columns if c not in ["facet_color", "facet"]]
            if facet_color_field:
                facet_tooltips.append(facet_color_field)

            if st.button("Reset All Views"):
                st.session_state.zoom_reset_key += 1
                st.rerun()

            # Master legend AT THE TOP.
            if facet_color_field:
                if is_numeric:
                    legend_chart = continuous_legend_chart(scheme=scheme, title=facet_color_by)
                    st.altair_chart(legend_chart, use_container_width=False)
                else:
                    categories = sorted(pd.unique(facet_df[facet_color_field]))
                    legend_chart = categorical_legend_chart(
                        categories=categories, scheme=scheme, title=facet_color_by, orient="horizontal"
                    )
                    st.altair_chart(legend_chart, use_container_width=True)

            # Create charts in a grid layout with shared zoom
            zoom = alt.selection_interval(bind="scales", name=f"zoom_selection_{st.session_state.zoom_reset_key}")
            
            st_cols = st.columns(facet_columns)
            for i, facet_value in enumerate(top_facets):
                with st_cols[i % facet_columns]:
                    chart_df = facet_df[facet_df["facet"] == facet_value]

                    chart = (
                        alt.Chart(chart_df, width=225, height=225)
                        .mark_circle(size=20, opacity=0.7)
                        .encode(
                            x=alt.X(x_col, scale=alt.Scale(domain=list(x_domain)), title=""),
                            y=alt.Y(y_col, scale=alt.Scale(domain=list(y_domain)), title=""),
                            color=color_enc,
                            tooltip=facet_tooltips,
                        )
                        .add_params(zoom)
                        .properties(title=alt.TitleParams(text=facet_value, anchor="middle"))
                    )
                    st.altair_chart(chart)

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

                fig_expr = go.Figure()
                fig_expr.add_trace(go.Scattergl(
                    x=expr_df.iloc[:, 0],
                    y=expr_df.iloc[:, 1],
                    mode='markers',
                    marker=dict(
                        color=expr_df["expression"],
                        colorscale=expr_palette,
                        showscale=True,
                        colorbar=dict(title='Expression')
                    ),
                    customdata=expr_df["expression"],
                    hovertemplate='%{customdata:.2f}<extra></extra>'
                ))
                fig_expr.update_layout(
                    width=1000,
                    height=1000,
                    xaxis_title=plot_df.columns[0],
                    yaxis_title=plot_df.columns[1],
                    xaxis_scaleanchor="y",
                    xaxis_scaleratio=1,
                    template="simple_white",
                    title=f"Expression of {selected_gene}"
                )
                st.plotly_chart(fig_expr)
            except Exception as exc:
                st.error(f"Could not plot expression for {selected_gene}: {exc}")

    with tab2:
        st.subheader("Cellular Composition Analysis")
        
        col1, col2 = st.columns(2)
        xaxis_col = col1.selectbox(
            "Group by (X-axis)", 
            options=categorical_cols, 
            index=0 if categorical_cols else -1,
            key="xaxis_col_altair"
        )
        stack_col = col2.selectbox(
            "Stack by (Composition)", 
            options=categorical_cols, 
            index=1 if len(categorical_cols) > 1 else 0,
            key="stack_col_altair"
        )

        normalize = st.checkbox("Normalize to 100%", value=True, key="norm_altair")
        
        if xaxis_col and stack_col:
            if xaxis_col == stack_col:
                st.warning("Please choose different categories for 'Group by' and 'Stack by'.")
            else:
                # Use Altair to create the stacked bar chart
                y_encoding = alt.Y('count()', stack='normalize', title="Percent (%)")
                if not normalize:
                    y_encoding = alt.Y('count()', stack=True, title="Cell Count")

                composition_chart = alt.Chart(filtered_adata.obs).mark_bar().encode(
                    x=alt.X(f"{xaxis_col}:N", title=xaxis_col),
                    y=y_encoding,
                    color=alt.Color(f"{stack_col}:N", title=stack_col)
                ).properties(
                    title=f"Composition of '{stack_col}' across '{xaxis_col}'"
                )
                
                st.altair_chart(composition_chart, use_container_width=True)
        else:
            st.info("Select categories to generate a composition plot.")

    with tab3:
        st.subheader("Differential Gene Expression Analysis")

        # UI for DGE controls
        dge_col = st.selectbox(
            "Compare groups within column:",
            options=categorical_cols,
            index=0 if categorical_cols else -1,
            key="dge_col"
        )
        
        if dge_col:
            groups = sorted(filtered_adata.obs[dge_col].astype(str).unique().tolist())
            if len(groups) < 2:
                st.warning(f"Column '{dge_col}' must have at least two groups to compare.")
            else:
                col1, col2 = st.columns(2)
                group1 = col1.selectbox("Group 1 (target):", options=groups, index=0)
                group2 = col2.selectbox("Group 2 (reference):", options=[g for g in groups if g != group1], index=1 if len(groups) > 1 else 0)

                method = st.selectbox("Method:", options=["t-test", "wilcoxon", "logreg"])

                if st.button("Run Analysis"):
                    if group1 == group2:
                        st.error("Group 1 and Group 2 must be different.")
                    else:
                        dge_results_df = run_dge(filtered_adata, dge_col, group1, group2, method)

                        st.session_state.dge_results = dge_results_df

    # Display DGE results if they exist in session state
    if "dge_results" in st.session_state:
        st.subheader("DGE Results")
        dge_results_df = st.session_state.dge_results

        # Volcano plot
        log2fc_col = 'logfoldchanges'
        pval_col = 'pvals_adj'
        
        # Add significance flags
        p_thresh = st.number_input("P-value threshold", value=0.05)
        fc_thresh = st.number_input("Log2 Fold-Change threshold", value=1.0)

        dge_results_df['-log10(p-val)'] = -np.log10(dge_results_df[pval_col] + 1e-300)
        dge_results_df['significant'] = 'Not significant'
        dge_results_df.loc[
            (dge_results_df['pvals_adj'] < p_thresh) & (dge_results_df[log2fc_col] > fc_thresh),
            'significant'
        ] = 'Up-regulated'
        dge_results_df.loc[
            (dge_results_df['pvals_adj'] < p_thresh) & (dge_results_df[log2fc_col] < -fc_thresh),
            'significant'
        ] = 'Down-regulated'
        
        volcano_fig = px.scatter(
            dge_results_df,
            x=log2fc_col,
            y='-log10(p-val)',
            color='significant',
            color_discrete_map={
                'Not significant': 'grey',
                'Up-regulated': 'red',
                'Down-regulated': 'blue'
            },
            hover_name='names',
            title=f"Volcano Plot: {group1} vs {group2}"
        )
        volcano_fig.add_vline(x=-fc_thresh, line_width=1, line_dash="dash", line_color="grey")
        volcano_fig.add_vline(x=fc_thresh, line_width=1, line_dash="dash", line_color="grey")
        volcano_fig.add_hline(y=-np.log10(p_thresh), line_width=1, line_dash="dash", line_color="grey")
        st.plotly_chart(volcano_fig, use_container_width=True)

        # Results table
        st.write("Differential expression results table:")
        st.dataframe(dge_results_df)

        st.subheader("Gene Set Enrichment Analysis (GSEA)")

        # Get significant genes from DGE results
        sig_genes = dge_results_df[dge_results_df['pvals_adj'] < p_thresh]['names'].tolist()

        if not sig_genes:
            st.warning("No significant genes found with the current thresholds. Cannot run GSEA.")
        else:
            st.write(f"Found **{len(sig_genes)}** significant genes to test for enrichment.")
            
            # GSEA controls
            col1, col2 = st.columns(2)
            gene_sets = col1.selectbox(
                "Gene Set Database",
                options=['GO_Biological_Process_2021', 'GO_Cellular_Component_2021', 'GO_Molecular_Function_2021', 'KEGG_2021_Human', 'Reactome_2022'],
            )
            organism = col2.selectbox("Organism", options=["Human", "Mouse"])

            if st.button("Run GSEA"):
                gsea_results_df = run_gsea(sig_genes, gene_sets, organism)
                st.session_state.gsea_results = gsea_results_df
    
    # Display GSEA results if they exist
    if "gsea_results" in st.session_state:
        st.subheader("GSEA Results")
        gsea_results_df = st.session_state.gsea_results

        if gsea_results_df.empty:
            st.warning("No enrichment results found for the given gene list and database.")
        else:
            # Bar chart of top terms
            gsea_results_df['-log10(p-val)'] = -np.log10(gsea_results_df['Adjusted P-value'])
            top_n = min(15, len(gsea_results_df))
            top_terms = gsea_results_df.nlargest(top_n, 'Combined Score')

            gsea_fig = px.bar(
                top_terms.sort_values('Combined Score', ascending=True),
                x='Combined Score',
                y='Term',
                orientation='h',
                title=f"Top {top_n} Enriched Terms in '{gene_sets}'",
                hover_data=['Adjusted P-value', 'Genes']
            )
            gsea_fig.update_layout(yaxis_title="")
            st.plotly_chart(gsea_fig, use_container_width=True)

            # GSEA results table
            st.write("Full GSEA results table:")
            st.dataframe(gsea_results_df)


    st.subheader("Metadata preview")
    st.write("First few rows from `obs`:")
    st.dataframe(plot_adata.obs.head(15))

    st.caption(
        "Tip: rerun with `streamlit run scrna_viewer_app.py -- --data-path /path/to/file.h5ad` "
        "to load a specific object without using the uploader."
    )


if __name__ == "__main__":
    main()
