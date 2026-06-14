# ============================================================
# questions.py  --  Generates interview questions using Mistral LLM
# ============================================================
# Uses the FREE Hugging Face Inference API with Mistral-7B.
# Falls back to HF_FALLBACK_MODEL (Zephyr) on 503, then to
# smart templates if both APIs are unavailable.
#
# Fixes applied:
#  #6  HF_FALLBACK_API_URL is now actually used (was defined but ignored)
#  #7  _parse_questions_from_text rejects non-question preamble lines
#      (lines >25 chars without "?" are now rejected unless they look
#       like a genuine question sentence)
# ============================================================

import re
import requests
from config import (
    HF_TOKEN, HF_API_URL, HF_FALLBACK_API_URL, HF_MODEL, HF_FALLBACK_MODEL,
    MAX_TOKENS_QUESTIONS, MAX_TOKENS_FOLLOWUP, DEFAULT_NUM_QUESTIONS,
    API_TIMEOUT, GEMMA_API_URL, USE_LOCAL_GEMMA
)


# ── Fallback template questions ─────────────────────────────────
SKILL_QUESTIONS = {
    "python": [
        "Can you walk me through a complex Python project you have built recently?",
        "How do you manage memory and performance in Python applications?",
        "Explain how you have used Python decorators or context managers in real projects.",
    ],
    "machine learning": [
        "Walk me through your end-to-end ML pipeline from data to deployment.",
        "How do you handle class imbalance and overfitting in your models?",
        "Describe a time when a model failed in production. How did you debug it?",
    ],
    "deep learning": [
        "What architecture choices did you make in your last deep learning project and why?",
        "How do you decide between CNNs, RNNs, and Transformers for a new problem?",
        "Explain how you debugged a training instability or vanishing gradient issue.",
    ],
    "nlp": [
        "What NLP tasks have you tackled such as NER, classification, or generation?",
        "Compare BERT and GPT from an architectural and use-case perspective.",
        "How have you fine-tuned a pre-trained language model for a custom task?",
    ],
    "aws": [
        "Describe a cloud architecture you have designed or improved on AWS.",
        "How do you handle cost optimisation and scaling on AWS?",
        "What CI/CD pipeline have you built for deploying ML models on AWS?",
    ],
    "flask": [
        "Describe the most complex REST API you have built with Flask.",
        "How do you handle authentication, rate limiting, and error handling in Flask?",
        "How did you scale and deploy a Flask app to production?",
    ],
    "sql": [
        "Describe a complex SQL query you wrote that improved performance significantly.",
        "How do you approach database schema design for a new application?",
        "What indexing strategies have you used to optimise slow queries?",
    ],
    "git": [
        "Describe the branching and release strategy your team uses with Git.",
        "How do you handle merge conflicts in a large team codebase?",
        "Walk me through your code review process.",
    ],
    "tensorflow": [
        "What custom layers or loss functions have you implemented in TensorFlow?",
        "How do you save, version, and serve TensorFlow models in production?",
    ],
    "pytorch": [
        "How have you used PyTorch autograd for custom training loops?",
        "Describe a custom dataset and DataLoader you implemented.",
    ],
}

GENERIC_QUESTIONS = [
    "Walk me through your most impactful project and what problem it solved.",
    "What is your biggest technical strength and how have you applied it professionally?",
    "Describe a time you had to learn a completely new technology quickly.",
    "Tell me about a difficult technical disagreement with a colleague and how you resolved it.",
    "What does your personal learning routine look like to stay current with technology?",
    "Describe a production incident you have dealt with and how you resolved it.",
]

WARMUP_QUESTION = (
    "Tell me about yourself, your background, what you have been working on recently, "
    "and what excites you most technically."
)
BEHAVIOURAL_QUESTION = (
    "Tell me about a time when you faced a significant challenge on a project under pressure. "
    "How did you handle it and what was the outcome?"
)

# Phrases that indicate an intro/preamble line rather than a question
_PREAMBLE_PATTERNS = re.compile(
    r"^(here are|sure|certainly|of course|below are|the following|"
    r"as requested|i('ve| have) generated|these questions|interview questions)",
    re.IGNORECASE
)


# ── Core LLM call helper ────────────────────────────────────────
def _call_hf(api_url, model_name, prompt, max_tokens=500):
    """
    Sends a prompt to a Hugging Face Inference API endpoint and returns
    the generated text.  Returns empty string on any failure.
    """
    if not HF_TOKEN or "paste_your_token" in HF_TOKEN:
        return ""

    headers = {"Authorization": "Bearer " + HF_TOKEN}
    payload = {
        "inputs": "[INST] " + prompt + " [/INST]",
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }
    }

    try:
        print("[AI] Calling " + model_name + "...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=API_TIMEOUT)

        # FIX #6: 503 = model still loading on HF free tier — caller should retry on fallback
        if response.status_code == 503:
            print("[WARN] " + model_name + " is loading (503) - will try fallback.")
            return None   # None signals "try fallback", "" signals "give up"

        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "").strip()
            print("[OK] " + model_name + " responded (" + str(len(text)) + " chars)")
            return text
        print("[WARN] Unexpected response format from " + model_name)
        return ""

    except requests.exceptions.Timeout:
        print("[WARN] " + model_name + " timed out.")
        return ""
    except Exception as e:
        print("[WARN] " + model_name + " error: " + str(e))
        return ""


def _call_mistral(prompt, max_tokens=500):
    """
    FIX #6: tries primary model first; on 503 retries with fallback model.
    Returns generated text string, or "" if both fail.
    """
    if not HF_TOKEN or "paste_your_token" in HF_TOKEN:
        print("[WARN] No HF_TOKEN set - using fallback template questions.")
        return ""

    result = _call_hf(HF_API_URL, HF_MODEL, prompt, max_tokens)
    if result is None:
        # Primary returned 503 — try fallback model
        print("[AI] Retrying with fallback model " + HF_FALLBACK_MODEL + "...")
        result = _call_hf(HF_FALLBACK_API_URL, HF_FALLBACK_MODEL, prompt, max_tokens)
        if result is None:
            result = ""
    return result


# ── Parse numbered list from LLM output ─────────────────────────
def _parse_questions_from_text(text):
    """
    Extracts a numbered list of questions from raw LLM output.

    FIX #7: previously, any line >25 chars was accepted even without "?",
    which let preamble sentences like "Here are 4 interview questions for..."
    pollute the list.  Now:
      - Lines must end with "?" OR start with a number/bullet marker.
      - Lines matching known preamble patterns are always rejected.
    """
    lines = text.strip().split("\n")
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Reject known preamble / intro lines
        if _PREAMBLE_PATTERNS.match(line):
            continue

        # Strip leading numbering / bullet markers
        cleaned = re.sub(r"^(\d+[\.\)]\s*|Q\d+[\.:]\s*|[-*]\s*)", "", line).strip()

        if not cleaned or len(cleaned) < 15:
            continue

        # FIX #7: require a "?" for lines without a numbered/bullet prefix;
        # numbered lines are accepted if they are long enough to be a question.
        has_marker   = bool(re.match(r"^\d+[\.\)]\s*", line))
        has_question = "?" in cleaned

        if has_question and len(cleaned) > 15:
            questions.append(cleaned)
        elif has_marker and len(cleaned) > 30:
            # Numbered line without "?" — only accept if it genuinely looks
            # like a question (contains a question verb near the start)
            question_verbs = re.compile(
                r"\b(describe|explain|how|what|walk|tell|can you|have you|"
                r"give|discuss|compare|why|when|which)\b", re.IGNORECASE
            )
            if question_verbs.search(cleaned[:60]):
                questions.append(cleaned)

    return questions


# ════════════════════════════════════════════════════════════════
# MAIN FUNCTION: Generate interview questions via Mistral
# ════════════════════════════════════════════════════════════════
def _generate_question_via_gemma(domain):
    """
    Calls the local fine-tuned Gemma API to generate an interview question for a specific domain.
    """
    if not USE_LOCAL_GEMMA:
        return None
    try:
        domain_title = domain.strip().title()
        instruction = f"Generate a {domain_title} interview question"
        payload = {
            "instruction": instruction,
            "input": "",
            "max_new_tokens": 150,
            "temperature": 0.4
        }
        response = requests.post(GEMMA_API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            completion = result.get("completion", "").strip()
            if completion:
                completion = re.sub(r'^["\']|["\']$', '', completion)
                return completion.strip()
    except Exception as e:
        print(f"[WARN] Local Gemma API call failed: {e}")
    return None


def generate_questions(candidate_skills, job_description, num_questions=DEFAULT_NUM_QUESTIONS):
    """
    Generates tailored interview questions using the local fine-tuned Gemma model when available,
    falling back to Mistral LLM or template questions.
    """
    # Guard: num_questions must be at least 3 (warm-up + 1 skill + behavioural)
    num_questions = max(num_questions, 3)
    skill_q_count = num_questions - 2   # slots between warm-up and behavioural

    skills_str = ", ".join(candidate_skills) if candidate_skills else "general software development"
    questions = [WARMUP_QUESTION]

    gemma_questions = []
    if USE_LOCAL_GEMMA and candidate_skills:
        print("[AI] Attempting to generate questions via local fine-tuned Gemma model...")
        for i in range(skill_q_count):
            skill = candidate_skills[i % len(candidate_skills)]
            q = _generate_question_via_gemma(skill)
            if q and q not in gemma_questions and q not in questions and q != BEHAVIOURAL_QUESTION:
                print(f"[OK] Gemma generated: {q}")
                gemma_questions.append(q)

    # Use Gemma questions if we generated enough
    if len(gemma_questions) >= skill_q_count:
        questions.extend(gemma_questions[:skill_q_count])
    else:
        # Otherwise, mix in what Gemma generated and request the rest from Mistral
        remaining_count = skill_q_count - len(gemma_questions)
        questions.extend(gemma_questions)

        print(f"[AI] Querying Mistral for remaining {remaining_count} questions...")
        prompt = (
            "You are an expert technical interviewer. Generate exactly "
            + str(remaining_count)
            + " interview questions for a candidate.\n\n"
            "Candidate skills: " + skills_str + "\n"
            "Job description: " + job_description + "\n\n"
            "Rules:\n"
            "- Questions must be specific and based on the candidate's actual skills\n"
            "- Questions should ask about real-world application, not just definitions\n"
            "- Each question must be on its own numbered line (1. 2. 3. etc.)\n"
            "- Do NOT include any introduction or conclusion - ONLY the numbered questions\n"
            "- Do NOT repeat similar questions\n\n"
            "Generate exactly " + str(remaining_count) + " questions now:"
        )

        llm_output    = _call_mistral(prompt, max_tokens=MAX_TOKENS_QUESTIONS)
        llm_questions = _parse_questions_from_text(llm_output) if llm_output else []

        if llm_questions:
            print("[OK] LLM generated " + str(len(llm_questions)) + " questions")
            for q in llm_questions:
                if len(questions) >= num_questions - 1:
                    break
                if q not in questions:
                    questions.append(q)
        else:
            print("[WARN] Using fallback template questions for remaining slots")
            _add_template_questions(candidate_skills, questions, skill_q_count)

    # Fill remaining skill-question slots from generics if any slots are still empty
    for q in GENERIC_QUESTIONS:
        if len(questions) >= num_questions - 1:
            break
        if q not in questions:
            questions.append(q)

    questions.append(BEHAVIOURAL_QUESTION)
    questions = questions[:num_questions]

    print("[INFO] Final question list: " + str(len(questions)) + " questions")
    return questions


def _add_template_questions(candidate_skills, questions, target_count):
    """Adds fallback template questions based on candidate skills."""
    seen = set(questions)
    for skill in candidate_skills:
        skill_lower = skill.lower()
        if skill_lower in SKILL_QUESTIONS:
            for q in SKILL_QUESTIONS[skill_lower]:
                if q not in seen and len(questions) < target_count:
                    questions.append(q)
                    seen.add(q)
    for q in GENERIC_QUESTIONS:
        if len(questions) >= target_count:
            break
        if q not in seen:
            questions.append(q)
            seen.add(q)


# ════════════════════════════════════════════════════════════════
# FOLLOW-UP: Generate a smart follow-up question
# ════════════════════════════════════════════════════════════════
def generate_followup(original_question, candidate_answer):
    """
    Generates a smart follow-up question based on the candidate's answer.
    """
    prompt = (
        "You are conducting a technical interview. The candidate just answered a question.\n\n"
        "Original question: " + original_question + "\n"
        "Candidate's answer: " + candidate_answer + "\n\n"
        "Generate ONE short, sharp follow-up question that:\n"
        "- Probes deeper into something specific the candidate mentioned\n"
        "- Asks for a concrete example or metric if they were vague\n"
        "- Challenges an assumption or asks about an edge case\n"
        "- Is no longer than 2 sentences\n\n"
        "Output ONLY the follow-up question, nothing else."
    )

    llm_output = _call_mistral(prompt, max_tokens=MAX_TOKENS_FOLLOWUP)

    if llm_output:
        lines = [l.strip() for l in llm_output.split("\n") if l.strip()]
        for line in lines:
            if _PREAMBLE_PATTERNS.match(line):
                continue
            cleaned = re.sub(r"^(\d+[\.\)]\s*|[-]\s*)", "", line).strip()
            if len(cleaned) > 10:
                return cleaned
        return llm_output.strip()

    # Fallback follow-ups
    answer_lower = candidate_answer.lower()
    if "built" in answer_lower or "created" in answer_lower or "developed" in answer_lower:
        return "What was the biggest technical challenge you faced while building that, and how specifically did you overcome it?"
    if "team" in answer_lower or "collaborated" in answer_lower:
        return "How did you handle disagreements or conflicting technical opinions within the team?"
    if "learned" in answer_lower or "studied" in answer_lower:
        return "Can you give a concrete example of how you applied that learning in a real project?"
    if "model" in answer_lower or "training" in answer_lower:
        return "What metrics did you use to measure the model performance, and what was the final result?"
    if "deployed" in answer_lower or "production" in answer_lower:
        return "What monitoring did you put in place after deployment, and did you encounter any incidents?"
    if "api" in answer_lower or "endpoint" in answer_lower:
        return "How did you handle authentication, rate limiting, and error cases in that API?"
    return "Can you walk me through a specific, concrete example that demonstrates that?"
