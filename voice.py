# ============================================================
# voice.py  --  Voice processing for Zara AI (Headless-safe)
# ============================================================
# Uses Whisper (offline) for client-uploaded audio transcription
# Uses gTTS to generate text-to-speech audio streams
# ============================================================

import os
from io import BytesIO
import whisper
from gtts import gTTS

# Load Whisper once at import time
print("Loading Whisper speech recognition model...")
whisper_model = whisper.load_model("base")
print("Whisper model ready!")


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
        result = whisper_model.transcribe(audio_file_path)
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
