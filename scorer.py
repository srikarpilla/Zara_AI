# ============================================================
# scorer.py  --  Evaluates candidate answers using Mistral LLM
# ============================================================
# Primary: Sends Q&A to Mistral for intelligent scoring (1-10)
# Fallback: Keyword + length heuristics if API is unavailable.
#
# Fixes applied:
#  #6  Uses HF_FALLBACK_API_URL on 503 (was always using primary URL)
#  #8  Heuristic score bands rebalanced so a short answer can't score 8+
#      just by hitting keyword count; length is now the dominant gate
#  #9  get_hire_recommendation score thresholds aligned with report.py
#      colour bands (≥8.5 Strong Hire, ≥7 Hire, ≥5.5 Consider, ≥4 Maybe)
#  #10 Question text in strengths/weaknesses uses textwrap.shorten so
#      truncated strings never end mid-word
# ============================================================

import re
import textwrap
import requests
from config import (
    HF_TOKEN, HF_API_URL, HF_FALLBACK_API_URL,
    HF_MODEL, HF_FALLBACK_MODEL,
    MAX_TOKENS_SCORING, API_TIMEOUT
)


# ── Internal LLM call (single endpoint) ────────────────────────
def _call_hf(api_url, model_name, prompt):
    headers = {"Authorization": "Bearer " + HF_TOKEN}
    payload = {
        "inputs": "[INST] " + prompt + " [/INST]",
        "parameters": {
            "max_new_tokens": MAX_TOKENS_SCORING,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=API_TIMEOUT)
        if response.status_code == 503:
            print("[WARN] " + model_name + " is loading (503).")
            return None   # None → try fallback
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return ""
    except Exception as e:
        print("[WARN] Scoring API error (" + model_name + "): " + str(e))
        return ""


# ── LLM-based scoring ──────────────────────────────────────────
def _score_with_llm(question, answer):
    """
    Sends a Q&A to Mistral (then Zephyr fallback) for a structured score.
    Returns dict with 'score' (int 1-10) and 'feedback' (str), or None on failure.
    """
    if not HF_TOKEN or "paste_your_token" in HF_TOKEN:
        return None

    prompt = (
        "You are an expert technical interviewer evaluating a candidate's answer.\n\n"
        "Question: " + question + "\n"
        "Candidate's Answer: " + answer + "\n\n"
        "Evaluate the answer on a scale of 1-10 using these criteria:\n"
        "- Technical accuracy and correctness\n"
        "- Depth and specificity (not just surface-level)\n"
        "- Use of concrete examples or real experience\n"
        "- Communication clarity\n"
        "- Completeness - did they fully address the question?\n\n"
        "Respond in EXACTLY this format (nothing else):\n"
        "SCORE: <number from 1 to 10>\n"
        "FEEDBACK: <2-3 sentences of specific, actionable feedback>"
    )

    # FIX #6: try primary, then fallback on 503
    text = _call_hf(HF_API_URL, HF_MODEL, prompt)
    if text is None:
        print("[AI] Scorer retrying with fallback model...")
        text = _call_hf(HF_FALLBACK_API_URL, HF_FALLBACK_MODEL, prompt)
        if text is None:
            text = ""

    return _parse_llm_score(text) if text else None


def _parse_llm_score(text):
    score_match    = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if not score_match:
        print("[WARN] Could not parse LLM score from: " + text[:80])
        return None

    score    = max(1, min(10, int(score_match.group(1))))
    feedback = feedback_match.group(1).strip() if feedback_match else "Good answer."

    sentences = re.split(r'(?<=[.!?])\s+', feedback)
    feedback  = " ".join(sentences[:3])

    return {"score": score, "feedback": feedback}


# ── Fallback heuristic scorer ───────────────────────────────────
def _score_heuristic(question, answer):
    """
    FIX #8: rebalanced so length is the primary gate.
    A short answer (<25 words) cannot score above 5 regardless of keywords.
    """
    answer_lower = answer.lower().strip()
    word_count   = len(answer.split())

    # ── Length band (primary gate) ────────────────────────────
    if word_count >= 120:
        length_score    = 7
        length_feedback = "Excellent level of detail."
    elif word_count >= 80:
        length_score    = 6
        length_feedback = "Good level of detail."
    elif word_count >= 40:
        length_score    = 5
        length_feedback = "Decent answer; more depth and examples would strengthen it."
    elif word_count >= 20:
        length_score    = 3
        length_feedback = "Answer was brief — try to elaborate with specifics."
    else:
        length_score    = 1
        length_feedback = "Answer was too short to evaluate properly."

    # ── Keyword bonus (secondary, max +2) ────────────────────
    technical_keywords = [
        "because", "example", "project", "used", "built", "implemented",
        "designed", "algorithm", "model", "data", "result", "performance",
        "solution", "approach", "experience", "framework", "library",
        "deployed", "tested", "optimized", "improved", "team", "challenge",
        "architecture", "database", "api", "production", "metric", "accuracy"
    ]
    keyword_hits = sum(1 for kw in technical_keywords if kw in answer_lower)

    if keyword_hits >= 6:
        kw_bonus    = 2
        kw_feedback = "Strong use of technical terminology and real-world context."
    elif keyword_hits >= 3:
        kw_bonus    = 1
        kw_feedback = "Some technical depth shown."
    else:
        kw_bonus    = 0
        kw_feedback = "Try to include more specific technical terms and concrete examples."

    # ── Vague language penalty ────────────────────────────────
    vague_phrases = ["i don't know", "not sure", "maybe", "i think so", "i guess", "i'm not certain"]
    if any(phrase in answer_lower for phrase in vague_phrases):
        vague_penalty   = 2
        vague_feedback  = "Answer lacked confidence — avoid phrases like 'I'm not sure'."
    else:
        vague_penalty   = 0
        vague_feedback  = ""

    score = max(1, min(10, length_score + kw_bonus - vague_penalty))

    parts = [length_feedback, kw_feedback]
    if vague_feedback:
        parts.append(vague_feedback)

    return {"score": score, "feedback": " ".join(parts)}


# ════════════════════════════════════════════════════════════════
# PUBLIC API: Score a single answer
# ════════════════════════════════════════════════════════════════
def score_single_answer(question, answer):
    """
    Tries LLM first, falls back to heuristics.
    Returns: {"score": int (1-10), "feedback": str, "method": "llm"|"heuristic"}
    """
    print("  Scoring answer (words: " + str(len(answer.split())) + ")...")

    llm_result = _score_with_llm(question, answer)
    if llm_result:
        llm_result["method"] = "llm"
        print("  [OK] LLM score: " + str(llm_result["score"]) + "/10")
        return llm_result

    result = _score_heuristic(question, answer)
    result["method"] = "heuristic"
    print("  [OK] Heuristic score: " + str(result["score"]) + "/10")
    return result


# ════════════════════════════════════════════════════════════════
# PUBLIC API: Evaluate all answers
# ════════════════════════════════════════════════════════════════
def evaluate_all_answers(interview_data):
    """
    Scores ALL Q&A pairs and returns a full evaluation dict.
    interview_data: list of {"question": str, "answer": str}
    """
    print("[SCORE] Evaluating all answers...")

    scored_results = []
    total_score    = 0

    for i, item in enumerate(interview_data):
        print("  Scoring answer " + str(i + 1) + "/" + str(len(interview_data)) + "...")
        result = score_single_answer(item["question"], item["answer"])
        scored_results.append({
            "question": item["question"],
            "answer":   item["answer"],
            "score":    result["score"],
            "feedback": result["feedback"],
            "method":   result.get("method", "heuristic")
        })
        total_score += result["score"]

    average_score  = round(total_score / len(interview_data), 1) if interview_data else 0
    recommendation = get_hire_recommendation(average_score, scored_results)

    print("[OK] Evaluation complete. Average: " + str(average_score) +
          "/10 (" + recommendation["verdict"] + ")")

    return {
        "results":         scored_results,
        "average_score":   average_score,
        "recommendation":  recommendation,
        "total_questions": len(interview_data)
    }


# ════════════════════════════════════════════════════════════════
# Hire recommendation
# ════════════════════════════════════════════════════════════════
def get_hire_recommendation(average_score, scored_results):
    """
    FIX #9: thresholds now match the colour bands in report.py
      ≥ 8.5  → Strong Hire  (green in report)
      ≥ 7.0  → Hire         (green in report)
      ≥ 5.5  → Consider     (yellow in report)
      ≥ 4.0  → Maybe        (yellow/red border)
      < 4.0  → No Hire      (red in report)
    """
    if average_score >= 8.5:
        verdict = "Strong Hire"
        summary = "Outstanding performance. The candidate demonstrated deep expertise and excellent communication."
    elif average_score >= 7.0:
        verdict = "Hire"
        summary = "Strong performance overall. The candidate is a solid fit for the role."
    elif average_score >= 5.5:
        verdict = "Consider"
        summary = "Mixed performance. The candidate shows potential but needs development in some areas."
    elif average_score >= 4.0:
        verdict = "Maybe"
        summary = "Below expectations. Recommend a second interview focusing on weaker areas."
    else:
        verdict = "No Hire"
        summary = "The candidate did not meet the technical bar for this role."

    sorted_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
    best_answer    = sorted_results[0]  if sorted_results else None
    worst_answer   = sorted_results[-1] if sorted_results else None

    # FIX #10: textwrap.shorten never cuts mid-word (adds "..." cleanly)
    def _shorten(text, width=70):
        return textwrap.shorten(text, width=width, placeholder="...")

    strengths = (
        "Best answer: \"" + _shorten(best_answer["question"]) +
        "\" (Score: " + str(best_answer["score"]) + "/10)"
        if best_answer else "N/A"
    )
    weaknesses = (
        "Needs improvement: \"" + _shorten(worst_answer["question"]) +
        "\" (Score: " + str(worst_answer["score"]) + "/10)"
        if worst_answer else "N/A"
    )

    return {
        "verdict":    verdict,
        "summary":    summary,
        "strengths":  strengths,
        "weaknesses": weaknesses
    }
