"""
IndustroSense AI - Streamlit Web Application
Three pages: Knowledge Retrieval & Agent | Multimodal Diagnosis | Governance & Model Card
Run locally with:  streamlit run app.py
"""
import streamlit as st
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import build_index, augmented_prompt
from agent import IndustroSenseAgent
from multimodal import fuse_modalities, caption_image, transcribe_speech
from security_governance import validate_text, validate_file, RateLimiter, AuditLogger, MODEL_CARD

st.set_page_config(page_title="IndustroSense AI", layout="wide")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "audit_log.jsonl")

if "store" not in st.session_state:
    st.session_state.store, st.session_state.docs, st.session_state.chunks = build_index(CORPUS_DIR)
    st.session_state.agent = IndustroSenseAgent(st.session_state.store)
    st.session_state.rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
    st.session_state.logger = AuditLogger(LOG_PATH)

st.sidebar.title("IndustroSense AI")
st.sidebar.caption("Multimodal Responsible GenAI for Industrial Diagnostics")
page = st.sidebar.radio("Module", [
    "1. Knowledge Retrieval & Agent",
    "2. Multimodal Diagnosis",
    "3. Governance & Model Card",
])
show_trace = st.sidebar.checkbox("Show sources / agent trace", value=True)

# ---------------------------------------------------------------- PAGE 1 ----
if page.startswith("1"):
    st.header("Knowledge Retrieval & Agent Module (RAG + Vector DB + AI Agent)")
    query = st.text_input("Enter a technical query about pump P-301 / P-305 faults:",
                           "Non-drive-end bearing is whining and vibration is rising")
    if st.button("Ask IndustroSense AI"):
        validation = validate_text(query)
        if not validation["valid"]:
            st.error(f"Input rejected: {validation['errors']}")
        elif not st.session_state.rate_limiter.allow("demo_user"):
            st.error("Rate limit exceeded. Please wait before sending another request.")
        else:
            result = st.session_state.agent.route(query)
            if result["clarifying_question"]:
                st.warning(result["clarifying_question"])
            else:
                st.success("Grounded answer composed from retrieved sources (see trace below).")
                if result["schedule_result"]:
                    st.metric("Schedule status", result["schedule_result"]["status"])
                    st.json(result["schedule_result"])
                for c in result["retrieved"]:
                    st.markdown(f"**{c['chunk_id']}**  (score={c['score']:.3f})")
                    st.write(c["text"])
            if show_trace:
                st.subheader("Agent decision trace")
                for step in result["trace"]:
                    st.text(f"{step['step']}: {step['content']}")
            st.session_state.logger.log("demo_user", query,
                                         [c["chunk_id"] for c in result["retrieved"]],
                                         "See trace", None, False)

# ---------------------------------------------------------------- PAGE 2 ----
elif page.startswith("2"):
    st.header("Multimodal Understanding & Generation Module")
    col1, col2 = st.columns(2)
    with col1:
        text_query = st.text_area("Text description of the fault:",
                                   "Pump P-301 non-drive-end bearing is whining, getting louder.")
        image_file = st.selectbox("Upload / select equipment image (simulated):",
                                   ["(none)", "bearing_housing_discolored.jpg", "suction_strainer_clogged.jpg"])
        audio_file = st.selectbox("Upload / select voice note (simulated):",
                                   ["(none)", "voice_note_1.wav", "voice_note_2.wav"])
    with col2:
        st.caption("Simulated stand-ins are used for vision/speech in this lab environment "
                   "(no network access to hosted VLM/ASR APIs). See Model Card, page 3.")

    if st.button("Run Multimodal Diagnosis"):
        img_tag = None if image_file == "(none)" else image_file
        aud_tag = None if audio_file == "(none)" else audio_file
        retrieved = st.session_state.store.search(text_query, k=2)
        fusion = fuse_modalities(text_query, img_tag, aud_tag, retrieved)
        st.write(f"**Modalities used:** {fusion['modalities_used']}")
        if fusion["image_result"]:
            st.write(f"**Image caption:** {fusion['image_result']['caption']} "
                     f"(confidence={fusion['image_result']['confidence']})")
        if fusion["speech_result"]:
            st.write(f"**Speech transcript:** {fusion['speech_result']['transcript']} "
                     f"(confidence={fusion['speech_result']['confidence']})")
        conf = fusion["overall_confidence"]
        st.metric("Overall confidence", conf)
        if fusion["needs_human_review"]:
            st.error("AI-Generated - Requires Human Review (confidence below 0.60)")
        else:
            st.success("Confidence acceptable for technician self-service use.")
        if show_trace:
            st.subheader("Fused context sent to generator")
            st.text(fusion["fused_context"])

# ---------------------------------------------------------------- PAGE 3 ----
else:
    st.header("Responsible Deployment & Governance")
    st.subheader("Model Card")
    for k, v in MODEL_CARD.items():
        st.markdown(f"**{k}**")
        if isinstance(v, list):
            for item in v:
                st.markdown(f"- {item}")
        else:
            st.write(v)
    st.subheader("Recent audit log entries")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            lines = f.readlines()[-10:]
        for line in lines:
            st.text(line.strip())
    else:
        st.info("No audit log entries yet - interact with modules 1 or 2 first.")
