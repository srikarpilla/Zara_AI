# ============================================================
# config.py  —  Central configuration for Zara AI
# ============================================================
# Put your Hugging Face token in a .env file:
#   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
#
# Get a FREE token at: https://huggingface.co/settings/tokens
#
# Fixes applied:
#  #5  API_TIMEOUT raised from 5 s to 30 s
#      (HF free-tier cold start takes 20-40 s; 5 s always timed out)
#  #6  HF_FALLBACK_API_URL exported so questions.py/scorer.py use it
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── Hugging Face ──────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HF_MODEL          = "mistralai/Mistral-7B-Instruct-v0.3"
HF_FALLBACK_MODEL = "HuggingFaceH4/zephyr-7b-beta"

HF_API_URL          = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_FALLBACK_API_URL = f"https://api-inference.huggingface.co/models/{HF_FALLBACK_MODEL}"  # FIX #6

# ── Interview ─────────────────────────────────────────────────
DEFAULT_NUM_QUESTIONS = 6
MAX_TOKENS_QUESTIONS  = 800
MAX_TOKENS_FOLLOWUP   = 200
MAX_TOKENS_SCORING    = 300

# FIX #5: raised from 5 s.  Set env var API_TIMEOUT=5 to restore
# instant-fallback behaviour for offline demos.
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "30"))

VOICE_RECORD_SECONDS = 90

# ── Local Gemma LLM (Fine-tuned for Question Generation) ───────
GEMMA_API_URL   = os.environ.get("GEMMA_API_URL", "http://127.0.0.1:8000/generate")
USE_LOCAL_GEMMA = os.environ.get("USE_LOCAL_GEMMA", "true").lower() == "true"

# ── Branding ──────────────────────────────────────────────────
APP_NAME    = "Zara AI"
APP_TAGLINE = "Your AI-Powered Interview Coach"
