
import io
import fnmatch
import zipfile
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Colors (RGBA)
TEST_COLOR  = "rgba(76, 114, 176, 0.9)"   # blue
TRAIN_COLOR = "rgba(221, 132, 82, 0.9)"   # orange
ANOM_COLOR  = "rgba(44, 160, 44, 0.35)"   # green fill
ANOM_LINE   = "rgba(44, 160, 44, 1.0)"    # green line


# -----------------------------
# Helpers
# -----------------------------
def _list_top_folders(zf: zipfile.ZipFile) -> List[str]:
    """Return top-level folder names in the zip (with trailing slash)."""
    folders = set()
    for name in zf.namelist():
        if "/" in name:
            top = name.split("/", 1)[0] + "/"
            folders.add(top)
    return sorted(folders)


def _find_labels_candidates(zf: zipfile.ZipFile) -> List[str]:
    return [n for n in zf.namelist() if n.lower().endswith("labels.csv")]


def _read_labels_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    # Expect columns: Name, Start, End
    if "Name" not in df.columns:
        raise ValueError("labels.csv must have a 'Name' column.")
    if "Start" not in df.columns or "End" not in df.columns:
        raise ValueError("labels.csv must have 'Start' and 'End' columns.")
    df = df.copy()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce").fillna(-1).astype(int)
    df["End"]   = pd.to_numeric(df["End"], errors="coerce").fillna(-1).astype(int)
    df.set_index("Name", inplace=True)
    return df


def _parse_series_name_and_test_start(file_name: str) -> Tuple[str, int]:
    """
    file_name example: "000_Anomaly_2500.csv"
    Returns: (name_without_ext, test_start)
    """
    base = file_name.rsplit("/", 1)[-1]          # strip folders if present
    name_no_ext = base.rsplit(".", 1)[0]
    splits = name_no_ext.split("_")
    test_start = int(splits[-1])
    return name_no_ext, test_start


def _read_series_from_zip(zf: zipfile.ZipFile, internal_path: str) -> np.ndarray:
    with zf.open(internal_path) as f:
        data = pd.read_csv(f, header=None).to_numpy().flatten()
    return data.astype(float)


def build_figure(data: np.ndarray, test_start: int, anomaly: Tuple[int, int], title: str) -> go.Figure:
    n = len(data)
    x_all = np.arange(n)

    fig = go.Figure()

    # Train
    if test_start > 0:
        fig.add_trace(
            go.Scatter(
                x=x_all[:test_start],
                y=data[:test_start],
                mode="lines",
                line=dict(width=1, color=TRAIN_COLOR),
                name="train",
                hovertemplate="t=%{x}<br>y=%{y}<extra></extra>",
            )
        )

    # Test
    if test_start < n:
        fig.add_trace(
            go.Scatter(
                x=x_all[test_start:],
                y=data[test_start:],
                mode="lines",
                line=dict(width=1, color=TEST_COLOR),
                name="test",
                hovertemplate="t=%{x}<br>y=%{y}<extra></extra>",
            )
        )

    a0, a1 = anomaly
    if a0 is not None and a1 is not None and a0 >= 0 and a1 > a0 and a1 <= n:
        # shaded region
        fig.add_vrect(
            x0=a0,
            x1=a1,
            fillcolor=ANOM_COLOR,
            line_width=0,
            layer="below",
        )
        # overlay line for clarity
        fig.add_trace(
            go.Scatter(
                x=x_all[a0:a1],
                y=data[a0:a1],
                mode="lines",
                line=dict(width=2, color=ANOM_LINE),
                name="anomaly",
                hovertemplate="ANOM t=%{x}<br>y=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title="t",
            showgrid=False,
            ticks="inside",
            rangeslider=dict(visible=True),
            type="linear",
        ),
        yaxis=dict(
            title="value",
            showgrid=False,
            ticks="inside",
        ),
        height=520,
    )
    return fig


def init_state():
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = None
    if "zip_name" not in st.session_state:
        st.session_state.zip_name = None
    if "folder_in_zip" not in st.session_state:
        st.session_state.folder_in_zip = None
    if "labels_df" not in st.session_state:
        st.session_state.labels_df = None  # original labels from file (if any)
    if "labels_overrides" not in st.session_state:
        st.session_state.labels_overrides = {}  # Dict[name, (start,end)]


def get_effective_anomaly(name: str, labels_df: Optional[pd.DataFrame]) -> Tuple[int, int]:
    # user override first
    if name in st.session_state.labels_overrides:
        return st.session_state.labels_overrides[name]
    # else from labels_df
    if labels_df is not None and name in labels_df.index:
        row = labels_df.loc[name]
        return int(row["Start"]), int(row["End"])
    return -1, -1


def set_override(name: str, start: int, end: int):
    st.session_state.labels_overrides[name] = (int(start), int(end))


def build_export_labels(series_names: List[str], labels_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in series_names:
        a0, a1 = get_effective_anomaly(name, labels_df)
        rows.append({"Name": name, "Start": int(a0), "End": int(a1)})
    out = pd.DataFrame(rows)
    return out


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Time Series Anomaly Labeler", layout="wide")
init_state()

st.title("Time Series Anomaly Labeler (Zip Upload + Interactive Plotly)")

with st.sidebar:
    st.header("1) Upload data")
    zup = st.file_uploader("Upload a zip file containing CSV time series (and optionally labels.csv)", type=["zip"])
    lup = st.file_uploader("Optional: upload labels.csv (overrides any labels.csv inside the zip)", type=["csv"])

    if zup is not None:
        st.session_state.zip_bytes = zup.getvalue()
        st.session_state.zip_name = zup.name

    # Load zip if present
    zf = None
    if st.session_state.zip_bytes is not None:
        try:
            zf = zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes))
        except zipfile.BadZipFile:
            st.error("That does not look like a valid zip file.")
            zf = None

    # Determine folder_in_zip and file list
    folder_in_zip = None
    series_files_internal = []
    series_files_display = []
    if zf is not None:
        folders = _list_top_folders(zf)
        # If there are folders, ask user which to use; else use root
        if folders:
            folder_in_zip = st.selectbox("Folder inside zip", folders, index=0)
        else:
            folder_in_zip = ""  # root

        # collect series files
        for name in zf.namelist():
            if folder_in_zip and not name.startswith(folder_in_zip):
                continue
            if fnmatch.fnmatch(name.lower(), "*.csv") and not name.lower().endswith("labels.csv"):
                series_files_internal.append(name)
                # display name stripped of folder prefix for readability
                display = name[len(folder_in_zip):] if folder_in_zip else name
                series_files_display.append(display)

        # Keep stable ordering
        order = np.argsort(series_files_display)
        series_files_internal = [series_files_internal[i] for i in order]
        series_files_display  = [series_files_display[i] for i in order]

        st.session_state.folder_in_zip = folder_in_zip

    st.divider()
    st.header("2) Labels source")
    labels_df = None

    # labels.csv from uploader
    if lup is not None:
        try:
            labels_df = _read_labels_from_bytes(lup.getvalue())
            st.success("Loaded labels.csv from upload.")
        except Exception as e:
            st.error(f"Could not read uploaded labels.csv: {e}")

    # else try to load labels.csv from zip
    if labels_df is None and zf is not None:
        candidates = _find_labels_candidates(zf)
        if candidates:
            # prefer labels.csv inside selected folder if possible
            preferred = None
            if st.session_state.folder_in_zip is not None:
                for c in candidates:
                    if c.startswith(st.session_state.folder_in_zip):
                        preferred = c
                        break
            pick = preferred or candidates[0]
            try:
                with zf.open(pick) as f:
                    labels_df = _read_labels_from_bytes(f.read())
                st.info(f"Loaded labels from zip: {pick}")
            except Exception as e:
                st.warning(f"Found labels.csv in zip but couldn't read it ({pick}): {e}")

    st.session_state.labels_df = labels_df

    st.divider()
    st.header("3) Export")
    export_name = st.text_input("Export filename", value="labels_updated.csv")
    st.caption("Exports current labels (original + your edits).")

# Main area
if zf is None:
    st.info("Upload a zip file in the sidebar to begin.")
    st.stop()

if not series_files_internal:
    st.warning("No CSV time series files found in the selected zip/folder (excluding labels.csv).")
    st.stop()

col_left, col_right = st.columns([0.35, 0.65], gap="large")

with col_left:
    st.subheader("Select & edit")
    idx = st.selectbox(
        "Time series file",
        options=list(range(len(series_files_display))),
        format_func=lambda i: series_files_display[i],
    )
    internal_path = series_files_internal[idx]
    display_name = series_files_display[idx]

    # Read series
    try:
        data = _read_series_from_zip(zf, internal_path)
        name, test_start = _parse_series_name_and_test_start(display_name)
    except Exception as e:
        st.error(f"Could not read {display_name}: {e}")
        st.stop()

    n = len(data)
    a0, a1 = get_effective_anomaly(name, st.session_state.labels_df)

    st.markdown(f"**Series:** `{name}`")
    st.markdown(f"**Length:** `{n}`")
    st.markdown(f"**Test starts at index:** `{test_start}`")

    has_key = f"has_anom_{name}"
    default_has = (a0 >= 0 and a1 > a0)
    if has_key not in st.session_state:
        st.session_state[has_key] = default_has

    has_anomaly = st.checkbox(
        "Has anomaly",
        key=has_key,
        help="Uncheck to remove anomaly label.",
    )
    if not has_anomaly:
        a0, a1 = -1, -1
        set_override(name, a0, a1)
        # Keep editor widget states consistent (if they exist)
        st.session_state[f"typed_start_{name}"] = 0
        st.session_state[f"typed_end_{name}"] = 1 if n > 0 else 0
        st.session_state[f"slider_{name}"] = (0, 1 if n > 0 else 0)
    else:
        # Clamp defaults to valid range
        if not (0 <= a0 < n):
            a0 = max(0, min(n - 1, a0 if a0 >= 0 else 0))
        if not (0 <= a1 <= n) or a1 <= a0:
            a1 = min(n, a0 + 1)

        st.markdown("**Edit anomaly boundaries**")

        # Keyboard inputs (precise)
        c_start, c_end, c_apply = st.columns([1, 1, 1])
        with c_start:
            typed_start = st.number_input(
                "Start",
                min_value=0,
                max_value=max(0, n - 1),
                value=int(a0),
                step=1,
                help="Start index (inclusive).",
                key=f"typed_start_{name}",
            )
        with c_end:
            typed_end = st.number_input(
                "End",
                min_value=1,
                max_value=n,
                value=int(a1),
                step=1,
                help="End index (exclusive). Must be > start.",
                key=f"typed_end_{name}",
            )
        with c_apply:
            st.write("")
            st.write("")
            apply_typed = st.button("Apply typed range", use_container_width=True)

        if apply_typed:
            if int(typed_end) <= int(typed_start):
                st.error("End must be greater than start.")
            else:
                set_override(name, int(typed_start), int(typed_end))
                st.rerun()

        # Slider (fast adjustments)
        rng = st.slider(
            "Anomaly range [start, end)",
            min_value=0,
            max_value=n,
            value=(int(a0), int(a1)),
            step=1,
            help="Drag the handles to adjust quickly. Use the number inputs above for exact values.",
            key=f"slider_{name}",
        )
        a0_new, a1_new = int(rng[0]), int(rng[1])
        if a1_new <= a0_new:
            st.warning("End must be greater than start.")
        else:
            # Update label immediately when slider moves
            set_override(name, a0_new, a1_new)
            a0, a1 = a0_new, a1_new

    # Quick buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Clear label"):
            set_override(name, -1, -1)
            st.session_state[has_key] = False
            st.session_state[f"typed_start_{name}"] = 0
            st.session_state[f"typed_end_{name}"] = 1 if n > 0 else 0
            st.session_state[f"slider_{name}"] = (0, 1 if n > 0 else 0)
            st.rerun()
    with b2:
        if st.button("Set to test window"):
            # Common heuristic: anomalies in test region
            start, end = int(test_start), int(n)
            set_override(name, start, end)
            # Ensure UI reflects this immediately
            st.session_state[has_key] = True
            st.session_state[f"typed_start_{name}"] = start
            st.session_state[f"typed_end_{name}"] = end
            st.session_state[f"slider_{name}"] = (start, end)
            st.rerun()
    with b3:
        if st.button("Reset to original"):
            if st.session_state.labels_df is not None and name in st.session_state.labels_df.index:
                row = st.session_state.labels_df.loc[name]
                start, end = int(row["Start"]), int(row["End"])
                set_override(name, start, end)
                st.session_state[has_key] = (start >= 0 and end > start)
                # Sync widgets
                st.session_state[f"typed_start_{name}"] = max(0, start) if start >= 0 else 0
                st.session_state[f"typed_end_{name}"] = max(1, end) if end > 0 else 1
                st.session_state[f"slider_{name}"] = (
                    int(max(0, start)) if start >= 0 else 0,
                    int(min(n, max(1, end))) if end > 0 else 1,
                )
            else:
                set_override(name, -1, -1)
                st.session_state[has_key] = False
                st.session_state[f"typed_start_{name}"] = 0
                st.session_state[f"typed_end_{name}"] = 1 if n > 0 else 0
                st.session_state[f"slider_{name}"] = (0, 1 if n > 0 else 0)
            st.rerun()

with col_right:
    st.subheader("Interactive plot")
    fig = build_figure(
        data=data,
        test_start=test_start,
        anomaly=(a0, a1),
        title="Train (orange) + Test (blue) with Anomaly (green)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True,
        },
    )
    st.caption("Tip: Use mouse wheel / trackpad to zoom, click-drag to pan, and the range slider to quickly navigate.")

# Export section (bottom)
st.divider()
st.subheader("Export labels")

# Create list of series "names" (without extension, without folder)
series_names = []
for disp in series_files_display:
    nm, _ = _parse_series_name_and_test_start(disp)
    series_names.append(nm)

export_df = build_export_labels(series_names, st.session_state.labels_df)

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download labels CSV",
    data=csv_bytes,
    file_name=export_name if export_name.endswith(".csv") else (export_name + ".csv"),
    mime="text/csv",
)
st.dataframe(export_df, use_container_width=True, height=240)
