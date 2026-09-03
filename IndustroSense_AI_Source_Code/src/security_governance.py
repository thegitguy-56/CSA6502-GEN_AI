"""
IndustroSense AI - Responsible Web Deployment & Governance Module
Implements input validation, a simple in-memory rate limiter, audit logging,
and a model-card generator, per Section D.7 and Part III of the assignment.
"""
import time
import json
import re

MAX_TEXT_LEN = 2000
MAX_IMAGE_BYTES = 8 * 1024 * 1024   # 8 MB
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a"}
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|previous|the) instructions",
    r"system prompt",
    r"reveal (your|the) (prompt|api key|credentials)",
    r"disregard (all|previous) rules",
]


def validate_text(text: str):
    errors = []
    if text is None or len(text.strip()) == 0:
        errors.append("Text query is empty.")
    elif len(text) > MAX_TEXT_LEN:
        errors.append(f"Text query exceeds max length {MAX_TEXT_LEN} chars.")
    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, text or "", re.IGNORECASE):
            errors.append(f"Potential prompt-injection pattern detected: '{pat}'.")
    return {"valid": len(errors) == 0, "errors": errors}


def validate_file(filename: str, size_bytes: int, kind: str):
    errors = []
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if kind == "image":
        if ext not in ALLOWED_IMAGE_EXT:
            errors.append(f"Disallowed image extension '{ext}'.")
        if size_bytes > MAX_IMAGE_BYTES:
            errors.append(f"Image exceeds max size {MAX_IMAGE_BYTES} bytes.")
    elif kind == "audio":
        if ext not in ALLOWED_AUDIO_EXT:
            errors.append(f"Disallowed audio extension '{ext}'.")
        if size_bytes > MAX_AUDIO_BYTES:
            errors.append(f"Audio exceeds max size {MAX_AUDIO_BYTES} bytes.")
    return {"valid": len(errors) == 0, "errors": errors}


class RateLimiter:
    """Simple fixed-window rate limiter: N requests per window_seconds per user."""

    def __init__(self, max_requests=5, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._log = {}  # user_id -> list[timestamps]

    def allow(self, user_id):
        now = time.time()
        history = [t for t in self._log.get(user_id, []) if now - t < self.window_seconds]
        allowed = len(history) < self.max_requests
        if allowed:
            history.append(now)
        self._log[user_id] = history
        return allowed


class AuditLogger:
    """Append-only structured audit log: prompt, retrieved sources, response, timestamp."""

    def __init__(self, path):
        self.path = path

    def log(self, user_id, query, retrieved_ids, response_summary, confidence, flagged):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_id": user_id,
            "query": query,
            "retrieved_chunk_ids": retrieved_ids,
            "response_summary": response_summary,
            "confidence": confidence,
            "flagged_for_human_review": flagged,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


MODEL_CARD = {
    "system_name": "IndustroSense AI",
    "version": "0.1.0 (lab prototype)",
    "intended_use": "Assist field engineers/technicians in diagnosing CP-2xx series centrifugal "
                     "feedwater pump faults using text, image, and voice fault reports, grounded in "
                     "the plant's manuals/SOPs/incident-log corpus.",
    "out_of_scope_use": "Not validated for equipment types outside the CP-2xx pump family; not a "
                         "substitute for a certified engineer's sign-off on safety-critical actions; "
                         "not for autonomous control of plant equipment.",
    "training_data": "No model was fine-tuned; the system uses retrieval over a fixed, versioned "
                      "document corpus (5 sample documents, see References) and simulated vision/speech "
                      "stand-ins documented in multimodal.py.",
    "known_limitations": [
        "Embeddings are a local TF-IDF+SVD substitute, not a hosted semantic embedding model; "
        "retrieval quality is weaker for paraphrased or synonym-heavy queries.",
        "Vision and speech modules are simulated stand-ins in this lab environment, not production VLM/ASR.",
        "No formal red-teaming has been performed; prompt-injection filtering is pattern-based only.",
    ],
    "human_oversight": "Outputs with overall confidence below 0.60 are auto-flagged "
                        "'AI-Generated - Requires Human Review' per governance policy POL-AI-003 Section 3.1.",
    "data_handling": "Uploaded images/voice notes are treated as operational data; retention and PII "
                      "handling follow POL-AI-003 Section 3.2 (not implemented as a retention job in this "
                      "lab prototype).",
}
