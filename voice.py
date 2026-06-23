# ============================================================
# voice.py  --  Voice processing for Zara AI (Headless-safe)
# ============================================================
# Uses Whisper (offline) for client-uploaded audio transcription
# Uses gTTS to generate text-to-speech audio streams
# ============================================================
# NOTE: Whisper is lazy-loaded on first transcription request to
# avoid OOM on Render's 512 MB free tier.  We use the "tiny"
# model (~150 MB RAM) instead of "base" (~1 GB RAM).
# ============================================================

import os
from io import BytesIO
from gtts import gTTS

# Lazy-load: model is None until the first call to transcribe_audio()
_whisper_model = None

def _get_whisper_model():
    """Return the shared Whisper model, loading it on first call."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[Whisper] Loading 'tiny' model (lazy)...")
        _whisper_model = whisper.load_model("tiny")
        print("[Whisper] Model ready.")
    return _whisper_model


def generate_tts_audio(text):
    """
    Generates an MP3 audio stream in memory for the given text using gTTS.
    Returns a BytesIO object.
    """
    print("Generating TTS for: " + text[:60] + "...")
    fp = BytesIO()
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(fp)
        fp.seek(0)
    except Exception as e:
        print("[ERROR] gTTS generation failed: " + str(e))
        # Fallback to an empty stream or raise
        fp = BytesIO()
    return fp


def transcribe_audio(audio_file_path):
    """
    Converts a recorded audio file into text using Whisper.
    Then deletes the temporary audio file.
    """
    print("Transcribing audio file: " + audio_file_path)
    if not os.path.exists(audio_file_path):
        print("[WARN] Audio file not found: " + audio_file_path)
        return ""

    try:
        result = _get_whisper_model().transcribe(audio_file_path)
        transcript = result["text"].strip() if result.get("text") else ""
        if transcript:
            print("Transcribed: " + transcript[:100] + "...")
        else:
            print("[WARN] Whisper returned empty transcription (silence).")
    except Exception as e:
        print("[ERROR] Whisper transcription failed: " + str(e))
        transcript = ""
    finally:
        try:
            os.remove(audio_file_path)
            print("Temporary audio file removed: " + audio_file_path)
        except Exception as e:
            print("[WARN] Could not remove temporary audio file: " + str(e))

    return transcript
