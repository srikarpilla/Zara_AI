# ============================================================
# app.py  --  Zara AI Interview Platform (Main Flask App)
# ============================================================
# Run with: python app.py
# Visit:    http://localhost:5000
#
# Fixes applied:
#  #1  secure_filename + PDF-only whitelist  → no path traversal on upload
#  #2  reports/ folder + session validation  → no arbitrary file reads on download
#  #3  server-side answer store              → no 4 KB session cookie overflow
# ============================================================

import os
import uuid
from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename   # FIX #1

from parser    import parse_resume
from questions import generate_questions, generate_followup
from voice     import transcribe_audio, generate_tts_audio
from scorer    import evaluate_all_answers
from report    import create_report
from config    import APP_NAME, APP_TAGLINE, DEFAULT_NUM_QUESTIONS
import tempfile


# ── Flask app setup ───────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "zara_ai_interview_secret_2025")

UPLOAD_FOLDER  = "uploads"
REPORTS_FOLDER = "reports"          # FIX #2
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}        # FIX #1: whitelist

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# FIX #3: server-side store — large answer data never touches the cookie
_interview_store: dict = {}


# ══════════════════════════════════════════════════════════════
# ROUTE 1: Home page
# ══════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html", app_name=APP_NAME, tagline=APP_TAGLINE)


# ══════════════════════════════════════════════════════════════
# ROUTE 2: Start the interview
# ══════════════════════════════════════════════════════════════
@app.route("/start_interview", methods=["POST"])
def start_interview():
    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    resume_file = request.files["resume"]

    # FIX #1: reject non-PDF before touching the filesystem
    if not resume_file.filename or not _allowed_file(resume_file.filename):
        return jsonify({"error": "Only PDF resumes are accepted"}), 400

    job_description = request.form.get("job_description", "Software Developer role")
    num_questions   = int(request.form.get("num_questions", DEFAULT_NUM_QUESTIONS))

    # FIX #1: strip path components; uuid prefix prevents collisions
    safe_name   = secure_filename(resume_file.filename)
    unique_name = str(uuid.uuid4()) + "_" + safe_name
    resume_path = os.path.join(UPLOAD_FOLDER, unique_name)
    resume_file.save(resume_path)
    print("[FILE] Resume saved: " + resume_path)

    candidate_info = parse_resume(resume_path)
    form_name      = request.form.get("candidate_name", "").strip()
    candidate_name = form_name if form_name else candidate_info["name"]
    print("[INFO] Candidate: " + candidate_name + " | Skills: " + str(candidate_info["skills"]))

    questions = generate_questions(
        candidate_skills=candidate_info["skills"],
        job_description=job_description,
        num_questions=num_questions
    )

    # FIX #3: only small scalars in the cookie; answers go server-side
    sid = str(uuid.uuid4())
    session["sid"]              = sid
    session["candidate_name"]   = candidate_name
    session["questions"]        = questions
    session["current_question"] = 0
    session["job_description"]  = job_description
    session["skills_found"]     = candidate_info["skills"]

    _interview_store[sid] = []

    first_question = questions[0]
    speech_text = "Hello " + candidate_name + "! Welcome to your Zara AI interview. Let's begin. " + first_question

    return jsonify({
        "status":          "started",
        "candidate_name":  candidate_name,
        "skills_found":    candidate_info["skills"],
        "total_questions": len(questions),
        "question_number": 1,
        "question":        first_question,
        "speech_text":     speech_text
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 3: Submit an answer (Accepts audio file or JSON text)
# ══════════════════════════════════════════════════════════════
@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    sid = session.get("sid")
    if not sid or sid not in _interview_store:
        return jsonify({"error": "No active interview session"}), 400

    candidate_answer = ""

    # Check if this is an audio file upload
    if "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file.filename == "":
            candidate_answer = "No answer provided."
        else:
            # Save the file temporarily to pass to Whisper
            suffix = os.path.splitext(secure_filename(audio_file.filename))[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_path = tmp.name
            tmp.close()

            try:
                audio_file.save(temp_path)
                candidate_answer = transcribe_audio(temp_path)
            except Exception as e:
                print("[ERROR] Transcription failed: " + str(e))
                candidate_answer = "Error transcribing audio."
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
    else:
        # Fallback to JSON or Form data (text mode)
        if request.is_json:
            candidate_answer = request.json.get("answer", "")
        else:
            candidate_answer = request.form.get("answer", "")

    if not candidate_answer or not candidate_answer.strip():
        candidate_answer = "No answer provided."

    questions       = session.get("questions", [])
    current_q_index = session.get("current_question", 0)

    if not questions or current_q_index >= len(questions):
        return jsonify({"error": "Question index out of range"}), 400

    current_question = questions[current_q_index]

    # FIX #3: store answers server-side
    _interview_store[sid].append({"question": current_question,
                                   "answer":   candidate_answer})

    next_q_index = current_q_index + 1

    if next_q_index >= len(questions):
        speech_text = "Thank you for your time. The interview is now complete. Zara AI will analyse your performance and generate your report."
        return jsonify({
            "status":  "completed",
            "message": "Interview complete! Generating your performance report...",
            "speech_text": speech_text,
            "your_answer_was": candidate_answer
        })

    next_question = questions[next_q_index]
    session["current_question"] = next_q_index

    return jsonify({
        "status":          "next_question",
        "question_number": next_q_index + 1,
        "total_questions": len(questions),
        "question":        next_question,
        "your_answer_was": candidate_answer,
        "speech_text":     next_question
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 3.2: API route for Text-to-Speech audio streaming
# ══════════════════════════════════════════════════════════════
@app.route("/api/tts")
def api_tts():
    text = request.args.get("text", "")
    if not text:
        return "Missing text", 400
    try:
        fp = generate_tts_audio(text)
        return send_file(fp, mimetype="audio/mp3")
    except Exception as e:
        print("[ERROR] TTS route error: " + str(e))
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ROUTE 3.5: Stop recording (Stub for backward compatibility)
# ══════════════════════════════════════════════════════════════
@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    return jsonify({"status": "stopping"})


# ══════════════════════════════════════════════════════════════
# ROUTE 4: Follow-up question
# ══════════════════════════════════════════════════════════════
@app.route("/get_followup", methods=["POST"])
def get_followup():
    answer          = request.json.get("answer", "")
    current_q_index = session.get("current_question", 0)
    questions       = session.get("questions", [])

    if not answer or not questions:
        return jsonify({"error": "Missing answer or session data"}), 400

    followup = generate_followup(questions[current_q_index], answer)
    return jsonify({"followup": followup})


# ══════════════════════════════════════════════════════════════
# ROUTE 5: Generate report
# ══════════════════════════════════════════════════════════════
@app.route("/generate_report", methods=["POST"])
def generate_report():
    sid            = session.get("sid")
    candidate_name = session.get("candidate_name", "Candidate")

    if not sid:
        return jsonify({"error": "No active interview session"}), 400

    interview_data = _interview_store.get(sid, [])
    if not interview_data:
        return jsonify({"error": "No interview data found"}), 400

    print("[START] Generating report for " + candidate_name + "...")
    evaluation = evaluate_all_answers(interview_data)

    # FIX #2: session-keyed name → unguessable; stored in reports/ folder only
    report_filename = "report_" + sid + ".pdf"
    report_path     = os.path.join(REPORTS_FOLDER, report_filename)
    create_report(candidate_name, evaluation, report_path)

    session["report_file"] = report_filename   # so download route can verify

    return jsonify({
        "status":         "report_ready",
        "average_score":  evaluation["average_score"],
        "recommendation": evaluation["recommendation"]["verdict"],
        "summary":        evaluation["recommendation"].get("summary", ""),
        "strengths":      evaluation["recommendation"].get("strengths", ""),
        "weaknesses":     evaluation["recommendation"].get("weaknesses", ""),
        "report_file":    report_filename,
        "results":        evaluation["results"]
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 6: Download report
# ══════════════════════════════════════════════════════════════
@app.route("/download_report/<filename>")
def download_report(filename):
    # FIX #2: the requested name must match what THIS session generated
    expected = session.get("report_file")
    if not expected or filename != expected:
        return jsonify({"error": "Report not found or not authorised"}), 403

    report_path = os.path.join(REPORTS_FOLDER, secure_filename(filename))
    if not os.path.isfile(report_path):
        return jsonify({"error": "Report file missing on server"}), 404

    return send_file(report_path, as_attachment=True)


# ══════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[START] " + APP_NAME + " -- " + APP_TAGLINE)
    print("[WEB] Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
