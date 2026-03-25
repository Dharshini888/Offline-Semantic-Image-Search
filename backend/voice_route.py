"""
Voice Search Route
==================
Add this endpoint to your main.py FastAPI app.

In main.py, either paste the route directly or:
    from voice_route import router as voice_router
    app.include_router(voice_router)
"""

import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from voice_engine import voice_engine

logger = logging.getLogger("VoiceRoute")
router = APIRouter()


@router.post("/voice_search")
async def voice_search(audio: UploadFile = File(...)):
    """
    Offline voice search endpoint.

    Accepts an audio recording from the browser (webm/ogg),
    transcribes it with Vosk (100% offline), and returns the text.

    The frontend then uses the text to run a normal /search query.

    Request:
        POST /voice_search
        Content-Type: multipart/form-data
        Body: audio file (webm, ogg, wav, mp3)

    Response:
        { "transcript": "dog playing in the park", "success": true }
        { "transcript": "",  "success": false, "error": "Model not loaded" }
    """
    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not voice_engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice engine not available. "
                "Install vosk (pip install vosk) and download a model from "
                "https://alphacephei.com/vosk/models into ../models/vosk-model-small-en-us"
            ),
        )

    # ── Read uploaded bytes ───────────────────────────────────────────────────
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received")

    # Determine suffix from content-type or filename
    content_type = audio.content_type or ""
    suffix_map = {
        "audio/webm":  ".webm",
        "audio/ogg":   ".ogg",
        "audio/wav":   ".wav",
        "audio/wave":  ".wav",
        "audio/mp4":   ".mp4",
        "audio/mpeg":  ".mp3",
        "audio/mp3":   ".mp3",
    }
    suffix = suffix_map.get(content_type, ".webm")
    if audio.filename:
        _, ext = os.path.splitext(audio.filename)
        if ext:
            suffix = ext

    # ── Transcribe ────────────────────────────────────────────────────────────
    logger.info(f"Transcribing {len(audio_bytes)} bytes ({suffix}) …")
    transcript = voice_engine.transcribe_bytes(audio_bytes, suffix=suffix)

    if not transcript:
        return {
            "transcript": "",
            "success":    False,
            "error":      "Could not transcribe audio. Try speaking more clearly.",
        }

    logger.info(f"Transcript: '{transcript}'")
    return {
        "transcript": transcript,
        "success":    True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION INSTRUCTIONS
# ──────────────────────────────────────────────────────────────────────────────
#
#  In your main.py add these two lines (near the top with other imports):
#
#      from voice_route import router as voice_router
#      app.include_router(voice_router)
#
#  That's it.  The endpoint will be available at POST /voice_search
#
#  Full offline pipeline:
#
#    Browser mic  →  MediaRecorder (webm)  →  POST /voice_search
#         ↓
#    ffmpeg converts webm → 16kHz mono WAV  (on your machine)
#         ↓
#    Vosk KaldiRecognizer processes WAV     (on your machine, no internet)
#         ↓
#    { "transcript": "dogs at the beach" }
#         ↓
#    Frontend runs POST /search with the transcript text
#