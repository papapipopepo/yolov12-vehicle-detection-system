import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
import tempfile
import cv2
import plotly.express as px


# Path setup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best.pt"

VEHICLE_CLASSES = ["bus", "car", "van"]


# Page config & custom CSS

st.set_page_config(
    page_title="Vehicle Detection — YOLO12",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e3a;
}

h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00f5c4, #7b5ea7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #c8c8e8 !important;
}

[data-testid="metric-container"] {
    background: #13131f;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: #00f5c4;
}
[data-testid="metric-container"] label {
    color: #6868a8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    color: #00f5c4 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00f5c4, #7b5ea7) !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

.stSlider [data-baseweb="slider"] {
    padding-top: 8px;
}

[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 1px dashed #2a2a4a !important;
    border-radius: 12px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-bottom: 1px solid #1e1e3a;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #6868a8 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #00f5c4 !important;
    background: #13131f !important;
    border-bottom: 2px solid #00f5c4 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    overflow: hidden;
}

.stDownloadButton > button {
    background: #13131f !important;
    color: #00f5c4 !important;
    border: 1px solid #00f5c4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 8px !important;
}

.density-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.density-LOW    { background: #0d2e1e; color: #00f5a0; border: 1px solid #00f5a0; }
.density-MEDIUM { background: #2e2200; color: #ffd700; border: 1px solid #ffd700; }
.density-HIGH   { background: #2e0d0d; color: #ff6b6b; border: 1px solid #ff6b6b; }

.stAlert { border-radius: 8px !important; border-left-width: 4px !important; }
hr { border-color: #1e1e3a !important; }
</style>
""", unsafe_allow_html=True)


# Session state

if "history" not in st.session_state:
    st.session_state.history = []


# Caching

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan ada file: models/best.pt"
        )
    return YOLO(str(MODEL_PATH))

@st.cache_resource
def get_annotators():
    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=6)
    return box, label

def detector_pipeline(image_bytes: bytes, model: YOLO, conf: float, iou: float):
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    results = model.predict(
        source=image_np,
        conf=conf,
        iou=iou,
        imgsz=1024,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(results).with_nms()

    class_names = detections.data.get("class_name", [])
    confidences = detections.confidence if detections.confidence is not None else []
    if len(class_names) == len(confidences):
        labels = [f"{c}  {p:.2f}" for c, p in zip(class_names, confidences)]
    else:
        labels = [str(c) for c in class_names]

    box_annotator, label_annotator = get_annotators()
    annotated = box_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    classcounts = dict(Counter(class_names))
    total = sum(classcounts.values())

    if total <= 3:
        density = "LOW"
    elif total <= 7:
        density = "MEDIUM"
    else:
        density = "HIGH"

    return annotated, classcounts, total, density


# Sidebar

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Semakin rendah = lebih banyak deteksi, tapi mungkin ada false positive")
    iou = st.slider("IoU (NMS) Threshold", 0.10, 0.95, 0.70, 0.05,
                    help="Mengontrol overlap bounding box yang diizinkan")

    st.divider()
    st.markdown("## 📋 Info Model")
    st.code("best.pt\nYOLO12 — bus · car · van", language="text")

    st.divider()
    st.markdown("## 📜 Riwayat Sesi")
    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} deteksi dilakukan")
        if st.button("🗑️ Reset History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Belum ada deteksi")


# Header

st.title("🚗 Vehicle Detection — YOLO12")
st.caption("Counting, Analysis & Traffic Density · Powered by YOLO12 + Supervision")
st.divider()


# Load model

try:
    with st.spinner("Memuat model..."):
        model = load_model()
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(str(e))
    st.stop()


# Tabs

tab_image, tab_video, tab_batch, tab_history = st.tabs([
    "🖼️  Single Image",
    "🎬  Video",
    "📦  Batch Upload",
    "📜  History"
])

# ──────────────────────────────
# TAB 1: Single Image
# ──────────────────────────────
with tab_image:
    uploaded_file = st.file_uploader(
        "Upload gambar kendaraan",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png", "webp"],
        key="single_upload"
    )

    if uploaded_file:
        col_prev, _ = st.columns([1, 1])
        with col_prev:
            st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

        if st.button("🔍 Deteksi Kendaraan", type="primary", key="btn_detect"):
            bytes_data = uploaded_file.getvalue()

            with st.spinner("Mendeteksi kendaraan..."):
                annotated, classcounts, total, density = detector_pipeline(
                    bytes_data, model, conf, iou
                )

            # Simpan ke history
            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Total": total,
                "Density": density,
                **{cls: classcounts.get(cls, 0) for cls in VEHICLE_CLASSES}
            })

            col_img, col_info = st.columns([3, 2])

            with col_img:
                st.subheader("Detection Result")
                st.image(annotated, caption="Detected Vehicles", use_container_width=True)

                buf = BytesIO()
                Image.fromarray(annotated).save(buf, format="PNG")
                st.download_button(
                    "📥 Download Hasil (PNG)",
                    data=buf.getvalue(),
                    file_name=f"result_{uploaded_file.name}",
                    mime="image/png"
                )

            with col_info:
                st.subheader("Summary")

                density_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.markdown(
                    f'<span class="density-badge density-{density}">'
                    f'{density_emoji[density]} {density}'
                    f'</span>',
                    unsafe_allow_html=True
                )
                st.markdown("")

                st.metric("Total Kendaraan", total)

                st.subheader("Counts per Class")
                if classcounts:
                    cols = st.columns(len(VEHICLE_CLASSES))
                    for i, cls in enumerate(VEHICLE_CLASSES):
                        cols[i].metric(cls.upper(), classcounts.get(cls, 0))
                else:
                    st.info("Tidak ada kendaraan terdeteksi.")

                if classcounts:
                    df_count = pd.DataFrame(
                        [(cls, classcounts.get(cls, 0)) for cls in VEHICLE_CLASSES],
                        columns=["Class", "Count"]
                    )
                    st.download_button(
                        "📊 Download CSV",
                        data=df_count.to_csv(index=False),
                        file_name="vehicle_counts.csv",
                        mime="text/csv"
                    )

            # Bar chart
            st.divider()
            st.subheader("📊 Distribusi Kendaraan")
            if classcounts:
                df_chart = pd.DataFrame(
                    [(cls, classcounts.get(cls, 0)) for cls in VEHICLE_CLASSES],
                    columns=["Class", "Count"]
                )
                fig_bar = px.bar(
                    df_chart,
                    x="Class",
                    y="Count",
                    color="Class",
                    color_discrete_sequence=["#00f5c4", "#7b5ea7", "#ff6b6b"],
                    labels={"Class": "Jenis", "Count": "Jumlah"},
                )
                fig_bar.update_layout(
                    paper_bgcolor="#13131f",
                    plot_bgcolor="#13131f",
                    font=dict(color="#c8c8e8", family="Space Mono"),
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                fig_bar.update_traces(marker_line_width=0)
                st.plotly_chart(fig_bar, use_container_width=True)

# ──────────────────────────────
# TAB 2: Video
# ──────────────────────────────
with tab_video:
    st.subheader("🎬 Vehicle Detection pada Video")
    st.caption("Upload video dan model akan memproses setiap frame.")

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )
    frame_skip = st.slider(
        "Proses setiap N frame", 1, 10, 3,
        help="Angka lebih besar = lebih cepat tapi kurang smooth"
    )

    if video_file:
        if st.button("▶️ Mulai Deteksi Video", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            stframe = st.empty()
            col_sv, col_ss = st.columns(2)
            progress = st.progress(0)
            stat_placeholder = col_sv.empty()
            density_placeholder = col_ss.empty()

            frame_idx = 0
            all_counts = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(
                        source=rgb, conf=conf, iou=iou,
                        imgsz=640, verbose=False
                    )[0]
                    detections = sv.Detections.from_ultralytics(results).with_nms()

                    class_names = detections.data.get("class_name", [])
                    confidences = detections.confidence if detections.confidence is not None else []
                    labels = (
                        [f"{c} {p:.2f}" for c, p in zip(class_names, confidences)]
                        if len(class_names) == len(confidences)
                        else [str(c) for c in class_names]
                    )

                    box_ann, lbl_ann = get_annotators()
                    annotated = box_ann.annotate(scene=rgb.copy(), detections=detections)
                    annotated = lbl_ann.annotate(scene=annotated, detections=detections, labels=labels)

                    classcounts = dict(Counter(class_names))
                    total = sum(classcounts.values())
                    all_counts.append(total)
                    density = "LOW" if total <= 3 else ("MEDIUM" if total <= 7 else "HIGH")

                    stframe.image(annotated, caption=f"Frame {frame_idx}", use_container_width=True)
                    stat_placeholder.metric("Kendaraan di Frame", total)
                    density_placeholder.metric("Density", density)

                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                frame_idx += 1

            cap.release()
            st.success(f"✅ Video selesai diproses! {frame_idx} frames.")

            if all_counts:
                st.subheader("📈 Grafik Jumlah Kendaraan per Frame")
                fig_line = px.line(
                    x=list(range(len(all_counts))),
                    y=all_counts,
                    labels={"x": "Frame", "y": "Jumlah Kendaraan"},
                    color_discrete_sequence=["#00f5c4"],
                )
                fig_line.update_layout(
                    paper_bgcolor="#13131f",
                    plot_bgcolor="#13131f",
                    font=dict(color="#c8c8e8", family="Space Mono"),
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_line, use_container_width=True)

# ──────────────────────────────
# TAB 3: Batch Upload
# ──────────────────────────────
with tab_batch:
    st.subheader("📦 Batch Processing")
    st.caption("Upload beberapa gambar sekaligus dan dapatkan ringkasan.")

    uploaded_files = st.file_uploader(
        "Upload Banyak Gambar",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "webp"],
        key="batch_upload"
    )

    if uploaded_files:
        st.info(f"📂 {len(uploaded_files)} gambar dipilih")

        if st.button("🔍 Proses Semua Gambar", type="primary"):
            batch_results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Memproses {file.name}..."):
                    annotated, classcounts, total, density = detector_pipeline(
                        file.getvalue(), model, conf, iou
                    )

                    batch_results.append({
                        "File": file.name,
                        "Total": total,
                        "Density": density,
                        **{cls: classcounts.get(cls, 0) for cls in VEHICLE_CLASSES}
                    })

                    st.session_state.history.append({
                        "Waktu": datetime.now().strftime("%H:%M:%S"),
                        "File": file.name,
                        "Total": total,
                        "Density": density,
                        **{cls: classcounts.get(cls, 0) for cls in VEHICLE_CLASSES}
                    })

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.success(f"✅ Selesai memproses {len(uploaded_files)} gambar!")

            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            st.subheader("📊 Ringkasan Batch")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rata-rata Kendaraan", f"{df_batch['Total'].mean():.1f}")
            c2.metric("Maks Kendaraan", df_batch['Total'].max())
            c3.metric("Density HIGH", (df_batch['Density'] == 'HIGH').sum())

            fig_batch = px.bar(
                df_batch,
                x="File",
                y="Total",
                color="Density",
                color_discrete_map={"LOW": "#00f5a0", "MEDIUM": "#ffd700", "HIGH": "#ff6b6b"},
                labels={"Total": "Jumlah Kendaraan"},
            )
            fig_batch.update_layout(
                paper_bgcolor="#13131f",
                plot_bgcolor="#13131f",
                font=dict(color="#c8c8e8", family="Space Mono"),
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_batch, use_container_width=True)

            st.download_button(
                "📥 Download Hasil Batch (CSV)",
                data=df_batch.to_csv(index=False),
                file_name="batch_results.csv",
                mime="text/csv"
            )

# ──────────────────────────────
# TAB 4: History
# ──────────────────────────────
with tab_history:
    st.subheader("📜 Riwayat Deteksi Sesi Ini")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history).fillna(0)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if len(df_hist) > 1:
            st.subheader("📈 Tren Total Kendaraan")
            fig_trend = px.line(
                df_hist,
                x=df_hist.index,
                y="Total",
                markers=True,
                color_discrete_sequence=["#00f5c4"],
                labels={"index": "Deteksi ke-", "Total": "Jumlah Kendaraan"},
            )
            fig_trend.update_layout(
                paper_bgcolor="#13131f",
                plot_bgcolor="#13131f",
                font=dict(color="#c8c8e8", family="Space Mono"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            fig_trend.update_traces(
                line=dict(width=2),
                marker=dict(size=8, color="#7b5ea7", line=dict(color="#00f5c4", width=2))
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.download_button(
            "📥 Export History (CSV)",
            data=df_hist.to_csv(index=False),
            file_name="detection_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Hapus Semua History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Belum ada riwayat deteksi. Coba upload gambar di tab Single Image atau Batch.")