# VSCode-1

## scRNA-seq viewer

An interactive Streamlit app for inspecting a prepped scRNA-seq AnnData (.h5ad) lives in `scrna_viewer_app.py`.

Run it with your prepped object:

```bash
streamlit run scrna_viewer_app.py -- --data-path /path/to/prepped_object.h5ad
```

Alternatively, launch without a path and use the sidebar uploader to point the app to a `.h5ad` file.
