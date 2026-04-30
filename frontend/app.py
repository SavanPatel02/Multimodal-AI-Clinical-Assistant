from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Clinical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sample files per modality (shown as quick-load hints in sidebar)
SAMPLE_FILES = {
    "Chest X-Ray": sorted(Path("data/raw/chest_xrays").glob("*.png")) if Path("data/raw/chest_xrays").exists() else [],
    "CT Scan":     sorted(Path("data/raw/ct_scans").glob("*.jpg")) if Path("data/raw/ct_scans").exists() else [],
    "MRI":         sorted(Path("data/raw/mri_scans").glob("*.jpg")) if Path("data/raw/mri_scans").exists() else [],
    "Ultrasound":  sorted(Path("data/raw/ultrasound").glob("*.jpg")) if Path("data/raw/ultrasound").exists() else [],
}
SAMPLE_AUDIO = sorted(Path("data/raw/audio_samples").glob("*.wav")) if Path("data/raw/audio_samples").exists() else []

# Session state for quick-loaded sample files
if "loaded_audio_path" not in st.session_state:
    st.session_state.loaded_audio_path = None
if "loaded_image_path" not in st.session_state:
    st.session_state.loaded_image_path = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    language = st.selectbox("Audio Language", ["en", "es", "fr", "de", "zh"], index=0)
    imaging_modality = st.selectbox(
        "Imaging Modality",
        ["Chest X-Ray", "CT Scan", "MRI", "Ultrasound"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Quick Load Sample Files**")
    st.caption("Click any file below to load it instantly — no upload needed.")

    # Audio samples
    if SAMPLE_AUDIO:
        with st.expander(f"🎙️ Audio ({len(SAMPLE_AUDIO)} files)"):
            for f in SAMPLE_AUDIO:
                label = f.stem.replace("_", " ").title()
                if st.button(label, key=f"aud_{f.name}", use_container_width=True):
                    st.session_state.loaded_audio_path = str(f)
                    st.rerun()
        if st.session_state.loaded_audio_path:
            st.success(f"Audio: {Path(st.session_state.loaded_audio_path).name}")
            if st.button("✕ Clear audio", key="clear_audio"):
                st.session_state.loaded_audio_path = None
                st.rerun()

    # Image samples for selected modality
    img_samples = SAMPLE_FILES.get(imaging_modality, [])
    if img_samples:
        with st.expander(f"🖼️ {imaging_modality} ({len(img_samples)} files)"):
            for f in img_samples:
                label = f.stem.replace("_", " ").title()
                if st.button(label, key=f"img_{f.name}", use_container_width=True):
                    st.session_state.loaded_image_path = str(f)
                    st.rerun()
        if st.session_state.loaded_image_path:
            st.success(f"Image: {Path(st.session_state.loaded_image_path).name}")
            if st.button("✕ Clear image", key="clear_image"):
                st.session_state.loaded_image_path = None
                st.rerun()
    else:
        st.warning(f"No {imaging_modality} samples found.\nRun: scripts/generate_modality_samples.py")

    st.markdown("---")
    st.markdown("**Model Stack**")
    st.markdown("- 🎙️ faster-whisper (CPU)")
    st.markdown("- 🫁 DenseNet121 + GradCAM (CPU)")
    st.markdown("- 🧠 Mistral-7B via Ollama (GPU)")
    st.markdown("- 📚 FAISS + PubMed RAG (CPU)")
    st.markdown("- 🔬 spaCy Clinical NER (CPU)")
    st.markdown("---")
    api_url = st.text_input("API Base URL", value=API_BASE)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Multimodal AI Clinical Assistant")
st.markdown(
    "Upload a voice recording and a medical image to generate a structured SOAP note "
    "with diagnosis suggestions and drug interaction warnings."
)
st.markdown("---")

# ── Upload columns ────────────────────────────────────────────────────────────
col_audio, col_image = st.columns(2)

# Resolve active audio: uploaded file takes priority, then session-state quick-load
def _read_path(path: str) -> io.BytesIO:
    buf = io.BytesIO(Path(path).read_bytes())
    buf.name = Path(path).name
    return buf

with col_audio:
    st.subheader("🎙️ Doctor's Voice Note")
    audio_file = st.file_uploader(
        "Upload audio  —  or use Quick Load from sidebar",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        key="audio_upload",
    )
    # Fall back to quick-loaded sample
    active_audio = audio_file
    if active_audio is None and st.session_state.loaded_audio_path:
        active_audio = _read_path(st.session_state.loaded_audio_path)
        st.info(f"Using sample: **{active_audio.name}**")

    if active_audio:
        st.audio(active_audio, format="audio/wav")
        active_audio.seek(0)
        audio_bytes = active_audio.read()
        active_audio.seek(0)
        try:
            import soundfile as sf
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            times = np.linspace(0, len(waveform) / sr, len(waveform))
            step = max(1, len(times) // 2000)
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(
                x=times[::step], y=waveform[::step],
                mode="lines", line=dict(color="#4CAF50", width=0.8),
                name="Waveform",
            ))
            fig_wave.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                height=200, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_wave, use_container_width=True)
        except Exception:
            pass

with col_image:
    st.subheader("🫁 Medical Image")
    image_file = st.file_uploader(
        "Upload image  —  or use Quick Load from sidebar",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="image_upload",
    )
    # Fall back to quick-loaded sample
    active_image = image_file
    if active_image is None and st.session_state.loaded_image_path:
        active_image = _read_path(st.session_state.loaded_image_path)
        st.info(f"Using sample: **{active_image.name}**")

    if active_image:
        active_image.seek(0)
        pil_img = Image.open(active_image)
        st.image(pil_img, caption=active_image.name, use_container_width=True)
        active_image.seek(0)

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────
run_btn = st.button("🚀 Analyze & Generate SOAP Note", type="primary", use_container_width=True)

if run_btn:
    if not active_audio or not active_image:
        st.error("Please upload or quick-load both an audio file and a medical image.")
        st.stop()

    with st.spinner("Running multimodal inference pipeline…"):
        t0 = time.time()
        try:
            active_audio.seek(0)
            active_image.seek(0)
            resp = requests.post(
                f"{api_url}/analyze",
                files={
                    "audio": (active_audio.name, active_audio, "audio/wav"),
                    "image": (active_image.name, active_image, "image/jpeg"),
                },
                data={"language": language, "imaging_modality": imaging_modality},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start the server with: `uvicorn src.api.main:app --reload`")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.success(f"Analysis complete in {elapsed:.1f}s")

    # ── Results ─────────────────────────────────────────────────────────────
    tab_soap, tab_vision, tab_ner, tab_refs, tab_raw = st.tabs(
        ["📋 SOAP Note", "🫁 Image Findings", "🔬 Entities", "📚 Literature", "🔧 Raw JSON"]
    )

    with tab_soap:
        soap = data.get("soap_note", {})
        st.markdown("### Transcription")
        st.info(data.get("transcription", ""))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🗣️ Subjective")
            st.write(soap.get("subjective", ""))
            st.markdown("#### 📊 Objective")
            st.write(soap.get("objective", ""))
        with col2:
            st.markdown("#### 🔍 Assessment")
            st.write(soap.get("assessment", ""))
            st.markdown("#### 💊 Plan")
            st.write(soap.get("plan", ""))

        st.markdown("---")
        col_dx, col_meds = st.columns(2)
        with col_dx:
            st.markdown("**Primary Diagnosis**")
            st.success(data.get("primary_diagnosis", "N/A"))
            ddx = data.get("differential_diagnoses", [])
            if ddx:
                st.markdown("**Differential Diagnoses**")
                for i, d in enumerate(ddx, 1):
                    st.write(f"{i}. {d}")
        with col_meds:
            meds = data.get("recommended_medications", [])
            if meds:
                st.markdown("**Recommended Medications**")
                for m in meds:
                    st.write(f"• {m}")
            st.markdown("**Follow-up**")
            st.write(data.get("follow_up", ""))

        drug_ixns = data.get("drug_interactions", [])
        if drug_ixns:
            st.markdown("---")
            st.markdown("### ⚠️ Drug Interaction Warnings")
            for ixn in drug_ixns:
                severity = ixn.get("severity", "unknown")
                color = {"major": "🔴", "moderate": "🟡", "minor": "🟢"}.get(severity, "⚪")
                with st.expander(f"{color} {ixn['drug_a']} × {ixn['drug_b']} ({severity})"):
                    st.write(ixn.get("description", ""))
                    st.markdown(f"**Recommendation:** {ixn.get('recommendation', '')}")

    with tab_vision:
        st.markdown(f"### Image Findings — {imaging_modality}")
        st.write(data.get("image_findings_summary", ""))

        top_findings = data.get("top_findings", [])
        if top_findings:
            st.markdown("**Top Detected Pathologies**")
            for f in top_findings:
                st.warning(f"• {f}")

        scores = data.get("pathology_scores", {})
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:10]
            labels = [s[0] for s in sorted_scores]
            values = [s[1] for s in sorted_scores]
            fig_bar = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=["#F44336" if v > 0.5 else "#FF9800" if v > 0.3 else "#4CAF50" for v in values],
            ))
            fig_bar.update_layout(
                title="Pathology Confidence Scores",
                xaxis_title="Probability", height=350,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if active_image:
            active_image.seek(0)
            col_orig, col_annotated = st.columns(2)
            with col_orig:
                st.markdown("**Original Image**")
                st.image(Image.open(active_image), use_container_width=True)
            with col_annotated:
                st.markdown("**GradCAM Heatmap — regions model focused on**")
                gradcam_b64 = data.get("gradcam_overlay_b64")
                if gradcam_b64:
                    import base64
                    img_bytes = base64.b64decode(gradcam_b64)
                    gradcam_img = Image.open(io.BytesIO(img_bytes))
                    top = top_findings[0] if top_findings else "top pathology"
                    st.image(gradcam_img, use_container_width=True,
                             caption=f"GradCAM for: {top} (red = high attention, tissue only)")
                else:
                    active_image.seek(0)
                    st.image(Image.open(active_image), use_container_width=True,
                             caption="GradCAM not available")

    with tab_ner:
        st.markdown("### Clinical Named Entity Recognition")
        entity_html = data.get("entity_html", "")
        if entity_html:
            st.markdown("**Highlighted SOAP Note**")
            st.markdown(
                f'<div style="line-height:2;padding:1em;border:1px solid #ccc;border-radius:6px">'
                f'{entity_html}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

        entities = data.get("named_entities", {})
        cols = st.columns(3)
        entity_types = [
            ("symptoms", "🤒 Symptoms"),
            ("diagnoses", "🔍 Diagnoses"),
            ("medications", "💊 Medications"),
            ("dosages", "⚖️ Dosages"),
            ("procedures", "🔧 Procedures"),
            ("anatomy", "🫀 Anatomy"),
        ]
        for i, (key, label) in enumerate(entity_types):
            with cols[i % 3]:
                items = entities.get(key, [])
                st.markdown(f"**{label}**")
                if items:
                    for item in items:
                        st.write(f"• {item}")
                else:
                    st.write("—")

    with tab_refs:
        st.markdown("### 📚 Retrieved PubMed Literature")
        refs = data.get("literature_references", [])
        if refs:
            for ref in refs:
                with st.expander(f"📄 {ref.get('title', 'Untitled')} (PMID: {ref.get('pmid', '?')})"):
                    st.markdown(f"**Relevance score:** {ref.get('score', 0):.4f}")
                    st.write(ref.get("text", ""))
        else:
            st.info("No literature retrieved. Build the PubMed FAISS index first.")

    with tab_raw:
        st.markdown("### Raw API Response")
        st.json(data)

        # Metrics
        st.markdown("### 📊 Inference Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Audio Duration", f"{data.get('audio_duration_s', 0):.1f}s")
        mc2.metric("Input Tokens", data.get("input_tokens", 0))
        mc3.metric("Output Tokens", data.get("output_tokens", 0))
        mc4.metric("Total Latency", f"{data.get('total_latency_s', 0):.2f}s")

        if data.get("run_id"):
            st.info(f"MLflow Run ID: `{data['run_id']}`")
