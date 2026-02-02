# app.py
import os
import fnmatch
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Time Series Anomaly Labeler", layout="wide")

# If you keep the uploaded files next to app.py, these defaults will work.
# Otherwise set them in the sidebar.
DEFAULT_LABELS_CSV = "labels.csv"
DEFAULT_ZIP_PATH = "phase_1.zip"
DEFAULT_FOLDER_IN_ZIP = "phase_1/"

test_color = "rgba(76, 114, 176, 0.9)"   # blue
train_color = "rgba(221, 132, 82, 0.9)"  # orange
anom_color = "rgba(44, 160, 44, 1.0)"    # green


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_locations(labels_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv_path)
    if "Name" not in df.columns or "Start" not in df.columns or "End" not in df.columns:
        raise ValueError("labels.csv must contain columns: Name, Start, End")
    df = df.copy()
    df.set_index("Name", inplace=True)
    return df


@st.cache_data(show_spinner=False)
def list_series_files(zip_path: str, folder_in_zip: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        files = np.sort([
            name[len(folder_in_zip):]
            for name in zf.namelist()
            if name.startswith(folder_in_zip)
            and fnmatch.fnmatch(name, "*.csv")
            and not name.endswith("labels.csv")
        ])
    return files


def parse_test_start_from_filename(file: str) -> int:
    # file like: "000_Anomaly_2500.csv"
    file_name = os.path.splitext(os.path.basename(file))[0]
    splits = file_name.split("_")
    return int(splits[-1])


@st.cache_data(show_spinner=False)
def read_series(zip_path: str, folder_in_zip: str, file: str) -> tuple[str, int, np.ndarray]:
    internal_name = folder_in_zip + file
    series_name = os.path.splitext(os.path.basename(file))[0]
    test_start = parse_test_start_from_filename(file)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(internal_name) as f:
            data = pd.read_csv(f, header=None).to_numpy().flatten().astype(float)

    return series_name, test_start, data


def build_figure(series_name: str, test_start: int, data: np.ndarray, anomaly: tuple[int, int]) -> go.Figure:
    x = np.arange(len(data))

    fig = go.Figure()

    # Train trace
    fig.add_trace(go.Scatter(
        x=x[:test_start],
        y=data[:test_start],
        mode="lines",
        line=dict(width=1, color=train_color),
        name="Train"
    ))

    # Test trace
    fig.add_trace(go.Scatter(
        x=x[test_start:],
        y=data[test_start:],
        mode="lines",
        line=dict(width=1, color=test_color),
        name="Test"
    ))

    # Test start marker
    fig.add_vline(
        x=test_start,
        line_width=1,
        line_dash="dot",
        line_color="gray",
        annotation_text="test start",
        annotation_position="top right",
    )

    # Anomaly highlight
    a0, a1 = anomaly
    if a0 is not None and a1 is not None and a0 >= 0 and a1 > a0:
        # line overlay
        fig.add_trace(go.Scatter(
            x=x[a0:a1],
            y=data[a0:a1],
            mode="lines",
            line=dict(width=2, color=anom_color),
            name="Anomaly"
        ))
        # shaded region
        fig.add_vrect(
            x0=a0, x1=a1,
            fillcolor="rgba(44,160,44,0.15)",
            line_width=0
        )

    fig.update_layout(
        title=f"{series_name} — Train/Test with Anomaly Labels",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Interactive zoom + range slider
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        type="linear",
        showgrid=False
    )
    fig.update_yaxes(showgrid=False)

    return fig


def clamp_anomaly(a0: int, a1: int, n: int) -> tuple[int, int]:
    if n <= 0:
        return -1, -1
    a0 = int(max(0, min(a0, n - 1)))
    a1 = int(max(0, min(a1, n)))
    if a1 <= a0:
        a1 = min(n, a0 + 1)
    return a0, a1


# -----------------------------
# UI
# -----------------------------
st.title("Time Series Anomaly Labeler")

with st.sidebar:
    st.header("Data sources")
    zip_path = st.text_input("ZIP path", value=DEFAULT_ZIP_PATH)
    folder_in_zip = st.text_input("Folder in ZIP", value=DEFAULT_FOLDER_IN_ZIP)
    labels_csv_path = st.text_input("labels.csv path", value=DEFAULT_LABELS_CSV)

    st.caption("Tip: keep `phase_1.zip` and `labels.csv` next to `app.py` or set paths here.")

# Validate inputs early
if not os.path.exists(zip_path):
    st.error(f"ZIP not found: {zip_path}")
    st.stop()

if not os.path.exists(labels_csv_path):
    st.error(f"labels.csv not found: {labels_csv_path}")
    st.stop()

try:
    locations = load_locations(labels_csv_path)
except Exception as e:
    st.error(f"Failed to read labels.csv: {e}")
    st.stop()

try:
    file_list = list_series_files(zip_path, folder_in_zip)
    if len(file_list) == 0:
        st.error("No series CSV files found inside the ZIP (check folder name).")
        st.stop()
except Exception as e:
    st.error(f"Failed to list series from ZIP: {e}")
    st.stop()

# Session state for editable labels
if "labels_map" not in st.session_state:
    # Initialize from labels.csv
    st.session_state.labels_map = {
        idx: (int(row["Start"]), int(row["End"]))
        for idx, row in locations.iterrows()
    }

# Select series
colA, colB = st.columns([2, 1])
with colA:
    selected_file = st.selectbox("Select a time series", file_list.tolist())
with colB:
    st.write("")
    st.write("")
    if st.button("Reset ALL edits to labels.csv"):
        st.session_state.labels_map = {
            idx: (int(row["Start"]), int(row["End"]))
            for idx, row in locations.iterrows()
        }
        st.success("Edits reset.")

series_name, test_start, data = read_series(zip_path, folder_in_zip, selected_file)

# Get current label (edited if present, else fallback)
curr = st.session_state.labels_map.get(series_name, (-1, -1))
curr_start, curr_end = int(curr[0]), int(curr[1])

# Editor controls
st.subheader("Edit anomaly annotation")

left, right = st.columns([1.2, 2.8], vertical_alignment="top")

with left:
    has_anomaly = st.checkbox("This series has an anomaly", value=(curr_start >= 0 and curr_end > curr_start))

    n = len(data)
    if has_anomaly:
        # Provide a single range slider for start/end
        # Ensure sensible defaults if missing
        if curr_start < 0 or curr_end <= curr_start:
            curr_start, curr_end = 0, min(n, max(1, n // 20))

        a0, a1 = st.slider(
            "Anomaly range (start, end)",
            min_value=0,
            max_value=max(1, n - 1),
            value=(int(curr_start), int(min(curr_end, n - 1))),
            help="Drag the handles to change the anomaly segment."
        )

        # End is exclusive in many labeling schemes; here we treat it as exclusive.
        # Make end at least start+1 and at most n.
        a0, a1 = clamp_anomaly(a0, a1 + 1, n)  # convert slider's inclusive end-ish to exclusive
        st.caption(f"Stored as Start={a0}, End={a1} (End is exclusive).")

        st.session_state.labels_map[series_name] = (a0, a1)
    else:
        st.session_state.labels_map[series_name] = (-1, -1)
        st.caption("Stored as Start=-1, End=-1 (no anomaly).")

    # Optional: quick numeric fine-tuning
    with st.expander("Fine-tune with exact numbers"):
        a0_in = st.number_input("Start (inclusive)", min_value=-1, max_value=n, value=int(st.session_state.labels_map[series_name][0]))
        a1_in = st.number_input("End (exclusive)", min_value=-1, max_value=n, value=int(st.session_state.labels_map[series_name][1]))
        if st.button("Apply exact values"):
            if a0_in < 0 or a1_in < 0:
                st.session_state.labels_map[series_name] = (-1, -1)
            else:
                st.session_state.labels_map[series_name] = clamp_anomaly(int(a0_in), int(a1_in), n)
            st.success("Applied.")

    if st.button("Reset this series to labels.csv"):
        if series_name in locations.index:
            row = locations.loc[series_name]
            st.session_state.labels_map[series_name] = (int(row["Start"]), int(row["End"]))
        else:
            st.session_state.labels_map[series_name] = (-1, -1)
        st.success("Reset series label.")

with right:
    anomaly = st.session_state.labels_map.get(series_name, (-1, -1))
    fig = build_figure(series_name, test_start, data, anomaly)
    st.plotly_chart(fig, use_container_width=True)

# Export section
st.subheader("Export labels")

# Build updated labels DataFrame:
# keep the union of original label rows + any new series labels created during session
all_names = sorted(set(locations.index.tolist()) | set(st.session_state.labels_map.keys()))
export_df = pd.DataFrame(
    {
        "Name": all_names,
        "Start": [int(st.session_state.labels_map.get(nm, (-1, -1))[0]) for nm in all_names],
        "End": [int(st.session_state.labels_map.get(nm, (-1, -1))[1]) for nm in all_names],
    }
)

col1, col2 = st.columns([1, 2])
with col1:
    st.download_button(
        label="Download labels_updated.csv",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="labels_updated.csv",
        mime="text/csv",
    )

with col2:
    st.dataframe(export_df, use_container_width=True, height=220)
