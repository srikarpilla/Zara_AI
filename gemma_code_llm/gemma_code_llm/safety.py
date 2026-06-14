from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SafetyFinding:
    name: str
    reason: str
    match: str


@dataclass(frozen=True)
class SafetyReport:
    blocked: bool
    findings: list[SafetyFinding]


_DANGEROUS_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "destructive_delete",
        "Destructive recursive deletion command found.",
        re.compile(r"\b(rm\s+-rf|Remove-Item\b.*\b-Recurse\b.*\b-Force\b|del\s+/f\s+/s\s+/q)\b", re.IGNORECASE),
    ),
    (
        "disk_format",
        "Disk formatting command found.",
        re.compile(r"\b(format\s+[a-z]:|mkfs\.[a-z0-9]+|diskpart)\b", re.IGNORECASE),
    ),
    (
        "secret_access",
        "Potential secret or credential access found.",
        re.compile(r"\b(AWS_SECRET_ACCESS_KEY|OPENAI_API_KEY|private_key|id_rsa)\b", re.IGNORECASE),
    ),
    (
        "reverse_shell",
        "Reverse shell pattern found.",
        re.compile(r"\b(nc\s+-e|bash\s+-i|/dev/tcp/|powershell\b.*-enc)\b", re.IGNORECASE),
    ),
]


def scan_text(text: str) -> SafetyReport:
    findings: list[SafetyFinding] = []
    for name, reason, pattern in _DANGEROUS_PATTERNS:
        for match in pattern.finditer(text or ""):
            findings.append(SafetyFinding(name=name, reason=reason, match=match.group(0)))
    return SafetyReport(blocked=bool(findings), findings=findings)


def assert_safe(text: str, *, enabled: bool = True) -> None:
    if not enabled:
        return

    report = scan_text(text)
    if not report.blocked:
        return

    details = "; ".join(f"{finding.name}: {finding.match}" for finding in report.findings)
    raise ValueError(f"Safety filter blocked this request: {details}")

