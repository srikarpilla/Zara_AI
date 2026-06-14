# ============================================================
# report.py  —  Creates a PDF report after the interview
# ============================================================
# Uses fpdf2 (free library) to build a clean PDF report.
#
# Fixes applied:
#  #9   Score colour bands now match scorer.py thresholds:
#         green  ≥ 7.0  (was ≥ 8.0  — inconsistent with "Hire" verdict)
#         yellow ≥ 5.5  (was ≥ 6.0)
#         red    < 5.5
#  #13  Unicode support: DejaVu font registered so Whisper transcripts
#       with non-Latin characters render properly instead of being
#       silently replaced with "?" via Latin-1 re-encoding.
#       Falls back gracefully to Helvetica if the font file is absent.
# ============================================================

import datetime
import os
from fpdf import FPDF


# ── Unicode font setup ────────────────────────────────────────
# DejaVuSans is bundled with many Linux/Mac installs and can be
# downloaded free from dejavu-fonts.github.io.
# Place DejaVuSans.ttf and DejaVuSans-Bold.ttf next to this file.
_FONT_DIR     = os.path.dirname(os.path.abspath(__file__))
_FONT_REGULAR = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
_FONT_BOLD    = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")
_USE_UNICODE  = os.path.isfile(_FONT_REGULAR) and os.path.isfile(_FONT_BOLD)

if _USE_UNICODE:
    print("[INFO] DejaVuSans font found — Unicode PDF rendering enabled.")
else:
    print("[INFO] DejaVuSans.ttf not found — falling back to Helvetica "
          "(non-Latin chars will be replaced). "
          "Download from https://dejavu-fonts.github.io/ for full Unicode support.")


def _safe(text, limit=None):
    """
    FIX #13: if Unicode font is loaded, no sanitisation needed.
    Without it, encode to Latin-1 replacing unmappable chars,
    then use textwrap.shorten for clean truncation.
    """
    import textwrap
    if limit:
        text = textwrap.shorten(str(text), width=limit, placeholder="...")
    if _USE_UNICODE:
        return text
    return text.encode("latin-1", errors="replace").decode("latin-1")


class InterviewReport(FPDF):
    """Custom FPDF subclass with branded header and footer."""

    def header(self):
        if _USE_UNICODE:
            self.set_font("DejaVu", "B", 14)
        else:
            self.set_font("Helvetica", "B", 14)
        self.set_text_color(50, 50, 150)
        self.cell(0, 10, "Zara AI - Interview Report", align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(50, 50, 150)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        if _USE_UNICODE:
            self.set_font("DejaVu", "", 8)
        else:
            self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Page " + str(self.page_no()), align="C")

    # ── Convenience helpers ───────────────────────────────────
    def set_body_font(self, bold=False, size=11):
        style = "B" if bold else ""
        if _USE_UNICODE:
            self.set_font("DejaVu", style, size)
        else:
            self.set_font("Helvetica", style, size)

    def heading(self, text, size=12, color=(0, 0, 0)):
        self.set_body_font(bold=True, size=size)
        self.set_text_color(*color)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text, size=11, color=(0, 0, 0)):
        self.set_body_font(bold=False, size=size)
        self.set_text_color(*color)
        self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")


def create_report(candidate_name, evaluation, output_path="interview_report.pdf"):
    """
    Creates a full PDF interview report.

    Parameters:
        candidate_name  → candidate's name
        evaluation      → result from scorer.evaluate_all_answers()
        output_path     → file path to write (should be inside reports/ folder)
    """
    print("Creating PDF report...")

    pdf = InterviewReport()

    # FIX #13: register Unicode font if available
    if _USE_UNICODE:
        pdf.add_font("DejaVu", "",  _FONT_REGULAR, uni=True)
        pdf.add_font("DejaVu", "B", _FONT_BOLD,    uni=True)

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Section 1: Candidate Info ─────────────────────────────
    pdf.heading(_safe("Candidate: " + candidate_name), size=16)

    today = datetime.date.today().strftime("%B %d, %Y")
    pdf.body(_safe("Interview Date: " + today),          size=11, color=(100, 100, 100))
    pdf.body("Total Questions: " + str(evaluation["total_questions"]),
             size=11, color=(100, 100, 100))
    pdf.ln(5)

    # ── Section 2: Overall Score ──────────────────────────────
    avg     = evaluation["average_score"]
    verdict = evaluation["recommendation"]["verdict"]

    # FIX #9: colour thresholds aligned with scorer.py verdicts
    #   green  ≥ 7.0 (Hire / Strong Hire)
    #   yellow ≥ 5.5 (Consider / Maybe)
    #   red    < 5.5 (No Hire)
    if avg >= 7.0:
        pdf.set_fill_color(200, 255, 200)
    elif avg >= 5.5:
        pdf.set_fill_color(255, 255, 200)
    else:
        pdf.set_fill_color(255, 200, 200)

    pdf.set_body_font(bold=True, size=14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 14,
             "  Overall Score: " + str(avg) + "/10    |    Result: " + verdict + "  ",
             fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Section 3: Summary ───────────────────────────────────
    rec = evaluation["recommendation"]
    pdf.body(_safe(rec.get("summary", "")), size=11, color=(60, 60, 60))
    pdf.ln(3)

    # ── Section 4: Strengths & Areas to Improve ──────────────
    pdf.heading("Strengths:", size=12, color=(0, 120, 0))
    pdf.body(_safe(rec.get("strengths", "N/A")))
    pdf.ln(3)

    pdf.heading("Areas to Improve:", size=12, color=(180, 0, 0))
    pdf.body(_safe(rec.get("weaknesses", "N/A")))
    pdf.ln(8)

    # ── Section 5: Question-by-Question Breakdown ────────────
    pdf.heading("Question-by-Question Breakdown", size=13, color=(50, 50, 150))
    pdf.set_draw_color(50, 50, 150)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    for i, item in enumerate(evaluation["results"]):
        # Question
        pdf.set_body_font(bold=True, size=11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, _safe("Q" + str(i + 1) + ": " + item["question"], limit=220),
                       new_x="LMARGIN", new_y="NEXT")

        # Answer preview
        pdf.set_body_font(bold=False, size=10)
        pdf.set_text_color(60, 60, 60)
        answer_text = _safe(item["answer"], limit=320)
        pdf.multi_cell(0, 7, "Answer: " + answer_text,
                       new_x="LMARGIN", new_y="NEXT")

        # Score + feedback (colour-coded per answer)
        score = item["score"]
        if score >= 8:
            pdf.set_text_color(0, 140, 0)
        elif score >= 5:
            pdf.set_text_color(180, 120, 0)
        else:
            pdf.set_text_color(200, 0, 0)

        pdf.set_body_font(bold=True, size=10)
        pdf.multi_cell(
            0, 7,
            "Score: " + str(score) + "/10  |  " + _safe(item["feedback"], limit=220),
            new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(5)

    pdf.output(output_path)
    print("Report saved to: " + output_path)
    return output_path
