"""
Offline Voice Engine using Vosk
================================
100% offline — no internet, no cloud APIs.
Audio is processed entirely on your machine.

Setup (one-time):
    pip install vosk pydub
    
    # Also install ffmpeg (needed to convert webm → wav):
    # Ubuntu/Debian:  sudo apt install ffmpeg
    # macOS:          brew install ffmpeg
    # Windows:        https://ffmpeg.org/download.html

    # Download Vosk model (small English, ~40MB):
    # https://alphacephei.com/vosk/models
    # Recommended: vosk-model-small-en-us-0.15
    # Place it at:  ../models/vosk-model-small-en-us

    # Larger / more accurate model (~1.8GB):
    # vosk-model-en-us-0.42-gigaspeech
"""

import os
import wave
import json
import logging
import tempfile
import subprocess

logger = logging.getLogger("VoiceEngine")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    Model = None
    KaldiRecognizer = None
    VOSK_AVAILABLE = False
    logger.warning(
        "vosk not installed. Run: pip install vosk\n"
        "Then download a model from https://alphacephei.com/vosk/models"
    )


# ── Constants ─────────────────────────────────────────────────────────────────
# Auto-detect best available Vosk model (large > medium > small)
def _find_best_vosk_model():
    import os as _os
    base = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", "models"))
    candidates = [
        "vosk-model-small-en-in-0.4",           # Indian English ~36MB ← best for Indian accents
        "vosk-model-en-us-0.22-lgraph",         # US English ~128MB
        "vosk-model-en-us-0.42-gigaspeech",     # US English ~1.8GB
        "vosk-model-small-en-us-0.15",          # US English small ~40MB
        "vosk-model-small-en-us",               # US English small (renamed)
    ]
    for c in candidates:
        path = _os.path.join(base, c)
        if _os.path.isdir(path):
            return path
    # Return default even if not found — error will be shown at load time
    return _os.path.join(base, "vosk-model-small-en-us")

DEFAULT_MODEL_PATH = _find_best_vosk_model()
SAMPLE_RATE        = 16000   # Vosk expects 16 kHz mono PCM
CHUNK_SIZE         = 4000    # frames per read chunk


class VoiceEngine:
    """
    Offline speech-to-text using Vosk.

    Usage:
        engine = VoiceEngine()
        text = engine.transcribe_upload("/tmp/recording.webm")
        # → "dog playing in the park"
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model      = None
        self.model_path = model_path
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _check_ffmpeg(self):
        """Log ffmpeg availability at startup."""
        try:
            r = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5
            )
            first_line = r.stdout.decode(errors="replace").split("\n")[0]
            logger.info(f"✅ ffmpeg found: {first_line}")
        except FileNotFoundError:
            logger.warning(
                "⚠️  ffmpeg NOT found in PATH — voice search will fail.\n"
                "Windows: download from https://www.gyan.dev/ffmpeg/builds/\n"
                "Extract to C:\\ffmpeg\\ and add C:\\ffmpeg\\bin to System PATH"
            )
        except Exception as e:
            logger.warning(f"ffmpeg check failed: {e}")

    def _load_model(self):
        self._check_ffmpeg()
        if not VOSK_AVAILABLE:
            logger.error("vosk package not found — voice search disabled.")
            return

        if not os.path.exists(self.model_path):
            logger.warning(
                f"Vosk model not found at '{self.model_path}'.\n"
                "Download from https://alphacephei.com/vosk/models and extract to that path."
            )
            return

        try:
            self.model = Model(self.model_path)
            logger.info(f"✅ Vosk model loaded from '{self.model_path}'")
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # ── Audio conversion ──────────────────────────────────────────────────────

    def _to_wav(self, input_path: str, output_path: str) -> bool:
        """
        Convert any audio format (webm, ogg, mp4, mp3, …) to
        16 kHz mono PCM WAV using ffmpeg.

        Returns True on success, False on failure.
        """
        # Find ffmpeg — checks PATH and common Windows install locations
        ffmpeg_bin = "ffmpeg"
        import shutil as _shutil
        if not _shutil.which("ffmpeg"):
            # Winget / Scoop / Chocolatey common locations on Windows
            _candidates = [
                # Exact WinGet install path (Gyan.FFmpeg)
                r"C:\Users\dhars\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
                r"C:\tools\ffmpeg\bin\ffmpeg.exe",
            ]
            import glob as _glob
            # Also search AppData where winget puts things
            _winget_glob = r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\*ffmpeg*\**\ffmpeg.exe"
            _candidates += _glob.glob(_winget_glob, recursive=True)
            for _c in _candidates:
                if os.path.exists(_c):
                    ffmpeg_bin = _c
                    logger.info(f"Found ffmpeg at: {_c}")
                    break
        cmd = [
            ffmpeg_bin, "-y",           # overwrite without asking
            "-i", input_path,           # input file (any format)
            "-ar", str(SAMPLE_RATE),    # resample to 16 kHz
            "-ac", "1",                 # mono
            "-f", "wav",                # output format
            output_path,
        ]
        # Rebuild cmd with more aggressive conversion flags for webm/opus from Chrome
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # force 16-bit PCM
            "-ar", str(SAMPLE_RATE),  # force 16000 Hz
            "-ac", "1",               # force mono
            "-f", "wav",
            output_path,
        ]
        logger.info(f"🔄 ffmpeg: converting {os.path.basename(input_path)} → WAV 16kHz mono PCM")
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if result.returncode != 0:
                err_msg = result.stderr.decode(errors="replace").strip()
                logger.error(f"❌ ffmpeg failed (exit {result.returncode}): {err_msg[:400]}")
                return False
            # Verify output file has content
            out_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            logger.info(f"✅ ffmpeg done — output WAV: {out_size} bytes")
            if out_size < 500:
                logger.warning("⚠️  WAV output very small — audio may be silent")
            return True
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found in PATH.\n"
                "Windows fix:\n"
                "  1. Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-essentials.zip)\n"
                "  2. Extract to C:\\ffmpeg\\\n"
                "  3. Add C:\\ffmpeg\\bin to System PATH\n"
                "  4. Restart the terminal and server"
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out during conversion")
            return False

    # ── Core transcription ────────────────────────────────────────────────────

    def _transcribe_wav(self, wav_path: str) -> str:
        """
        Transcribe a 16 kHz mono WAV file with Vosk.
        Returns the recognised text string (empty string on failure).
        """
        if not self.is_ready:
            logger.warning("Vosk model not loaded — cannot transcribe")
            return ""

        try:
            with wave.open(wav_path, "rb") as wf:
                actual_channels = wf.getnchannels()
                actual_rate     = wf.getframerate()
                sampwidth       = wf.getsampwidth()
                n_frames        = wf.getnframes()
                raw_data        = wf.readframes(n_frames)

            logger.info(f"🔊 WAV: {actual_rate}Hz, {actual_channels}ch, {sampwidth*8}bit, {n_frames} frames")

            # Convert raw bytes to numpy int16 array
            import numpy as _np
            audio = _np.frombuffer(raw_data, dtype=_np.int16)

            # Mix down to mono if stereo
            if actual_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(_np.int16)
            elif actual_channels > 2:
                audio = audio.reshape(-1, actual_channels)[:, 0].astype(_np.int16)

            # Resample to 16000 Hz if needed (linear interpolation — fast, good enough for speech)
            if actual_rate != SAMPLE_RATE:
                logger.info(f"🔄 Resampling {actual_rate}Hz → {SAMPLE_RATE}Hz")
                orig_len  = len(audio)
                target_len = int(orig_len * SAMPLE_RATE / actual_rate)
                indices   = _np.linspace(0, orig_len - 1, target_len)
                audio     = _np.interp(indices, _np.arange(orig_len), audio).astype(_np.int16)
                logger.info(f"✅ Resampled: {orig_len} → {len(audio)} samples")

            # Feed to Vosk in chunks
            rec = KaldiRecognizer(self.model, SAMPLE_RATE)
            rec.SetWords(True)
            full_text_parts = []
            chunk = 8192
            for i in range(0, len(audio), chunk):
                block = audio[i:i+chunk].tobytes()
                if rec.AcceptWaveform(block):
                    partial = json.loads(rec.Result())
                    if partial.get("text"):
                        full_text_parts.append(partial["text"])

            # Flush remaining
            final = json.loads(rec.FinalResult())
            if final.get("text"):
                full_text_parts.append(final["text"])

            result_text = " ".join(full_text_parts).strip()
            if result_text:
                logger.info(f"✅ Vosk transcript: '{result_text}'")
            else:
                logger.warning("⚠️  Vosk returned empty — no speech detected in audio")
            return result_text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe_upload(self, audio_path: str) -> str:
        """
        Transcribe an uploaded audio file (webm, ogg, wav, mp3, …).

        Steps:
          1. If not already WAV, convert with ffmpeg → temp WAV
          2. Run Vosk transcription
          3. Clean up temp file
          4. Return text

        Args:
            audio_path: path to saved audio file

        Returns:
            Transcribed text, e.g. "dog playing in the park"
            Empty string if transcription failed or model not loaded.
        """
        if not self.is_ready:
            return ""

        # If it's already a 16 kHz mono WAV we can use it directly
        if audio_path.lower().endswith(".wav"):
            try:
                with wave.open(audio_path, "rb") as wf:
                    if wf.getnchannels() == 1 and wf.getframerate() == SAMPLE_RATE:
                        return self._transcribe_wav(audio_path)
            except Exception:
                pass  # fall through to ffmpeg conversion

        # Convert to WAV via ffmpeg
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        try:
            ok = self._to_wav(audio_path, tmp_wav.name)
            if not ok:
                return ""
            return self._transcribe_wav(tmp_wav.name)
        finally:
            try:
                os.unlink(tmp_wav.name)
            except OSError:
                pass

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
        """
        Transcribe raw audio bytes (e.g. from a FastAPI UploadFile).
        """
        logger.info(f"🎤 Received audio: {len(audio_bytes)} bytes, format={suffix}")
        if len(audio_bytes) < 500:
            logger.warning(f"⚠️  Audio too small ({len(audio_bytes)} bytes) — recording may be empty. Hold mic button longer.")
            return ""
        # Save a copy for debugging (remove in production)
        try:
            debug_path = os.path.join(os.path.dirname(__file__), "..", "data", f"last_recording{suffix}")
            with open(debug_path, "wb") as _df:
                _df.write(audio_bytes)
            logger.info(f"💾 Saved debug audio to {debug_path}")
        except Exception:
            pass
        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            tmp_in.close()
            # Browser sends pre-converted 16kHz mono WAV — skip ffmpeg entirely
            if suffix.lower() == ".wav":
                logger.info("📄 WAV from browser — going direct to Vosk (no ffmpeg)")
                return self._transcribe_wav(tmp_in.name)
            return self.transcribe_upload(tmp_in.name)
        finally:
            try:
                os.unlink(tmp_in.name)
            except OSError:
                pass


# ── Global singleton ──────────────────────────────────────────────────────────
voice_engine = VoiceEngine()