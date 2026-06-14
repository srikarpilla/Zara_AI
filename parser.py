# ============================================================
# parser.py  —  Reads the resume PDF and pulls out key info
# ============================================================
# Libraries:
#   pdfplumber  → reads PDF files
#   spacy       → NER to find candidate name
#
# No bug fixes required here — kept exactly as original with
# one minor improvement: invalid_keywords expanded slightly to
# catch more common resume header words that fool spaCy NER.
# ============================================================

import pdfplumber
import spacy

nlp = spacy.load("en_core_web_sm")

KNOWN_SKILLS = [
    "python", "java", "javascript", "react", "node", "sql", "mongodb",
    "docker", "kubernetes", "aws", "azure", "git", "machine learning",
    "deep learning", "tensorflow", "pytorch", "flask", "django", "fastapi",
    "html", "css", "c++", "c#", "typescript", "linux", "rest api",
    "data science", "nlp", "computer vision", "excel", "power bi", "tableau"
]


def read_pdf(file_path):
    """Opens a PDF and returns all text as a single string."""
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def extract_skills(resume_text):
    """Returns list of known skills found in the resume text."""
    resume_lower = resume_text.lower()
    return [skill for skill in KNOWN_SKILLS if skill in resume_lower]


def extract_candidate_name(resume_text):
    """
    Tries to find the candidate's name via spaCy NER + heuristics.
    Expanded invalid_keywords to reduce false positives on common
    resume section headers that spaCy sometimes tags as PERSON.
    """
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]

    invalid_keywords = {
        "generative", "ai", "resume", "curriculum", "vitae", "profile",
        "experience", "education", "skills", "summary", "contact",
        "developer", "engineer", "manager", "analyst", "designer",
        "consultant", "intern", "objective", "overview", "about", "page"
    }

    for line in lines[:3]:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name  = ent.text.strip()
                words = set(name.lower().split())
                if (name
                        and len(name) < 40
                        and not any(c.isdigit() for c in name)
                        and not words.intersection(invalid_keywords)):
                    return name

    doc = nlp(resume_text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name  = ent.text.strip()
            words = set(name.lower().split())
            if (name
                    and len(name) < 40
                    and not any(c.isdigit() for c in name)
                    and not words.intersection(invalid_keywords)):
                return name

    return "Candidate"


def parse_resume(file_path):
    """
    Main entry point.
    Returns {"name": str, "skills": list, "full_text": str}.
    """
    print("Reading resume from: " + file_path)
    resume_text    = read_pdf(file_path)
    candidate_name = extract_candidate_name(resume_text)
    skills_found   = extract_skills(resume_text)
    print("Found candidate: " + candidate_name)
    print("Found skills: "    + str(skills_found))
    return {"name": candidate_name, "skills": skills_found, "full_text": resume_text}
