"""
IndustroSense AI - Multimodal Understanding & Generation Module
Covers: text, image, speech intake -> per-modality processing -> late fusion
with RAG-retrieved knowledge -> consolidated diagnostic response.

Environment note: this lab sandbox has no network access to hosted
vision-language or ASR model endpoints (e.g., GPT-4V, Whisper API). Per the
assignment's explicit allowance to use "real or simulated" models, the
image-captioning and speech-to-text functions below are SIMULATED using
lightweight, deterministic keyword-extraction stand-ins that consume the same
inputs/outputs a real VLM/ASR call would (image bytes -> caption string;
audio bytes -> transcript string). Swapping in a real API only requires
replacing the body of `caption_image()` / `transcribe_speech()`.
"""
import hashlib

# Simulated vision-language captioner: maps an image "fingerprint" (here, a
# filename/metadata tag standing in for real image bytes) to a caption,
# mimicking what a VLM would output for equipment-fault photos.
_SIMULATED_IMAGE_CAPTIONS = {
    "bearing_housing_discolored.jpg":
        "Close-up of a pump bearing housing. Grease visible at the seal is dark brown/black in colour. "
        "Housing surface shows no visible cracks. Mild rust staining near the mounting bolt.",
    "suction_strainer_clogged.jpg":
        "Photograph of a suction strainer basket partially covered in fibrous debris and scale buildup, "
        "reducing open flow area.",
}


def caption_image(image_tag: str) -> dict:
    caption = _SIMULATED_IMAGE_CAPTIONS.get(
        image_tag,
        "Photograph of a mechanical component; no specific fault-signature keywords matched in the "
        "simulated caption bank."
    )
    conf = 0.81 if image_tag in _SIMULATED_IMAGE_CAPTIONS else 0.40
    return {"modality": "image", "input": image_tag, "caption": caption, "confidence": conf}


_SIMULATED_AUDIO_TRANSCRIPTS = {
    "voice_note_1.wav":
        "There's a whining noise coming from the non drive end bearing on pump three oh one, "
        "it's gotten louder over the last two shifts.",
    "voice_note_2.wav":
        "I hear a crackling sound near the suction side of pump three oh five and the discharge "
        "pressure gauge looks low.",
}


def transcribe_speech(audio_tag: str) -> dict:
    transcript = _SIMULATED_AUDIO_TRANSCRIPTS.get(
        audio_tag, "[simulated ASR could not confidently transcribe audio tag: unknown sample]"
    )
    conf = 0.88 if audio_tag in _SIMULATED_AUDIO_TRANSCRIPTS else 0.30
    return {"modality": "speech", "input": audio_tag, "transcript": transcript, "confidence": conf}


def fuse_modalities(text_query, image_tag, audio_tag, retrieved_chunks):
    """Late fusion: run each modality's model independently, convert every
    modality to a text representation, then concatenate with retrieved
    knowledge into a single grounded context for the response generator.
    Overall confidence = min() across modalities (weakest-link, conservative)."""
    parts = []
    confidences = []

    if text_query:
        parts.append(f"[TEXT QUERY]: {text_query}")
        confidences.append(0.95)  # user-typed text assumed reliable

    img_result = None
    if image_tag:
        img_result = caption_image(image_tag)
        parts.append(f"[IMAGE CAPTION]: {img_result['caption']}")
        confidences.append(img_result["confidence"])

    speech_result = None
    if audio_tag:
        speech_result = transcribe_speech(audio_tag)
        parts.append(f"[SPEECH TRANSCRIPT]: {speech_result['transcript']}")
        confidences.append(speech_result["confidence"])

    for i, c in enumerate(retrieved_chunks, 1):
        parts.append(f"[RETRIEVED-{i} | {c['doc_id']}]: {c['text'][:200]}...")

    fused_context = "\n".join(parts)
    overall_confidence = round(min(confidences), 2) if confidences else 0.0
    needs_human_review = overall_confidence < 0.6

    return {
        "fused_context": fused_context,
        "image_result": img_result,
        "speech_result": speech_result,
        "overall_confidence": overall_confidence,
        "needs_human_review": needs_human_review,
        "modalities_used": [m for m, present in
                             [("text", bool(text_query)), ("image", bool(image_tag)), ("speech", bool(audio_tag))]
                             if present],
    }
