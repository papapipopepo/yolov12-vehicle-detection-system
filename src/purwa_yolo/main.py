import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import supervision as sv
from PIL import Image
from ultralytics import YOLO

# --- OpenCV import guard (biar errornya jelas) ---
try:
    import cv2
except Exception as e:
    st.error("OpenCV (cv2) gagal di-load. Pastikan pakai opencv-python-headless dan numpy kompatibel.")
    st.exception(e)
    st.stop()

import requests

# =========================
# Google Drive Model Config
# =========================


MODEL_URL = os.getenv("MODEL_URL", "").strip()
# MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "").strip()
MODEL_GDRIVE_ID = "1Hq-Vlz5R1jWTs6OH1wtHnc8TVkkykn-L"
if not MODEL_GDRIVE_ID:
    # kalau user hanya isi URL, ambil file_id dari URL
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL atau MODEL_GDRIVE_ID belum di-set. Isi di .env atau Streamlit Secrets.")
    MODEL_GDRIVE_ID = extract_gdrive_file_id(MODEL_URL)

if not MODEL_GDRIVE_ID:
    raise RuntimeError("Gagal mengambil Google Drive file_id. Pastikan MODEL_URL valid.")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best.pt"

VEHICLE_CLASSES = ["bus", "car", "van"]


def extract_gdrive_file_id(url: str) -> str | None:
    """
    Support:
    - https://drive.google.com/file/d/<ID>/view?...
    - https://drive.google.com/open?id=<ID>
    - https://drive.google.com/uc?id=<ID>&export=download
    """
    patterns = [
        r"drive\.google\.com/file/d/([^/]+)/",
        r"[?&]id=([^&]+)",
        r"drive\.google\.com/uc\?id=([^&]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def download_gdrive_file(file_id: str, destination: Path, chunk_size: int = 32768) -> None:
    """
    Download Google Drive file (handles large file confirm token).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    def get_confirm_token(resp: requests.Response) -> str | None:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    # 1st request
    resp = session.get(base_url, params={"id": file_id}, stream=True)
    token = get_confirm_token(resp)

    # If token exists, confirm
    if token:
        resp.close()
        resp = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)

    resp.raise_for_status()

    tmp_path = destination.with_suffix(".part")
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    # Basic sanity check: .pt file should not be tiny HTML
    if tmp_path.stat().st_size < 200_000:  # 200 KB
        # This usually means we downloaded an HTML page instead of the model
        with open(tmp_path, "rb") as f:
            head = f.read(200).lower()
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Download gagal: file terlalu kecil (kemungkinan HTML, bukan .pt). "
            "Cek permission Google Drive (harus 'Anyone with the link can view')."
        )

    tmp_path.replace(destination)


@st.cache_resource
def load_model():
    MODELS_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        st.info("Downloading model dari Google Drive (first run)...")
        download_gdrive_file(MODEL_GDRIVE_ID, MODEL_PATH)

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


# =========================
# UI (Page config & CSS)
# =========================
st.set_page_config(
    page_title="Vehicle Detection — YOLO12",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
section[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e3a; }
h1 {
    font-weight: 800 !important;
    font-size: 2rem !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00f5c4, #7b5ea7, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
[data-testid="metric-container"] {
    background: #13131f; border: 1px solid #1e1e3a; border-radius: 12px; padding: 16px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00f5c4, #7b5ea7) !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    text-transform: uppercase;
}
.density-badge {
    display: inline-block; padding: 6px 20px; border-radius: 100px;
    font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.1rem;
    letter-spacing: 2px; text-transform: uppercase;
}
.density-LOW    { background: #0d2e1e; color: #00f5a0; border: 1px solid #00f5a0; }
.density-MEDIUM { background: #2e2200; color: #ffd700; border: 1px solid #ffd700; }
.density-HIGH   { background: #2e0d0d; color: #ff6b6b; border: 1px solid #ff6b6b; }
hr { border-color: #1e1e3a !important; }
</style>
""", unsafe_allow_html=True)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU (NMS) Threshold", 0.10, 0.95, 0.70, 0.05)

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

# TAB 1: Single Image
with tab_image:
    uploaded_file = st.file_uploader(
        "Upload gambar kendaraan",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png", "webp"],
        key="single_upload"
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

        if st.button("🔍 Deteksi Kendaraan", type="primary", key="btn_detect"):
            bytes_data = uploaded_file.getvalue()

            with st.spinner("Mendeteksi kendaraan..."):
                annotated, classcounts, total, density = detector_pipeline(bytes_data, model, conf, iou)

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
                    f'<span class="density-badge density-{density}">{density_emoji[density]} {density}</span>',
                    unsafe_allow_html=True
                )
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

            st.divider()
            st.subheader("📊 Distribusi Kendaraan")
            if classcounts:
                df_chart = pd.DataFrame(
                    [(cls, classcounts.get(cls, 0)) for cls in VEHICLE_CLASSES],
                    columns=["Class", "Count"]
                )
                fig_bar = px.bar(df_chart, x="Class", y="Count", color="Class", labels={"Class": "Jenis", "Count": "Jumlah"})
                st.plotly_chart(fig_bar, use_container_width=True)

# TAB 2: Video
with tab_video:
    st.subheader("🎬 Vehicle Detection pada Video")
    st.caption("Upload video dan model akan memproses setiap frame.")

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video_upload")
    frame_skip = st.slider("Proses setiap N frame", 1, 10, 3)

    if video_file and st.button("▶️ Mulai Deteksi Video", type="primary"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        stframe = st.empty()
        col_sv, col_ss = st.columns(2)
        progress = st.progress(0.0)
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
                results = model.predict(source=rgb, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
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

            progress.progress(min(frame_idx / total_frames, 1.0))
            frame_idx += 1

        cap.release()
        st.success(f"✅ Video selesai diproses! {frame_idx} frames.")

        if all_counts:
            st.subheader("📈 Grafik Jumlah Kendaraan per Frame")
            fig_line = px.line(x=list(range(len(all_counts))), y=all_counts, labels={"x": "Frame", "y": "Jumlah Kendaraan"})
            st.plotly_chart(fig_line, use_container_width=True)

# TAB 3: Batch Upload
with tab_batch:
    st.subheader("📦 Batch Processing")
    uploaded_files = st.file_uploader(
        "Upload Banyak Gambar",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "webp"],
        key="batch_upload"
    )

    if uploaded_files and st.button("🔍 Proses Semua Gambar", type="primary"):
        batch_results = []
        progress_bar = st.progress(0.0)

        for i, file in enumerate(uploaded_files):
            annotated, classcounts, total, density = detector_pipeline(file.getvalue(), model, conf, iou)

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

        df_batch = pd.DataFrame(batch_results)
        st.success(f"✅ Selesai memproses {len(uploaded_files)} gambar!")
        st.dataframe(df_batch, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Hasil Batch (CSV)",
            data=df_batch.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv"
        )

# TAB 4: History
with tab_history:
    st.subheader("📜 Riwayat Deteksi Sesi Ini")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history).fillna(0)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

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