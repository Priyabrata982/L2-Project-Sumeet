"""PII masking utilities to keep personal data out of prompts and logs."""

from __future__ import annotations

import re

PII_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("card_number", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[MASKED_CARD]"),
    ("aadhaar_id", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[MASKED_ID]"),
    ("pan_id", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[MASKED_ID]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[MASKED_EMAIL]"),
    (
        "phone",
        re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4,5}\b"),
        "[MASKED_PHONE]",
    ),
    (
        "address",
        re.compile(
            r"\b\d{1,5}\s+[A-Za-z0-9.\s]{2,30}\b(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Sector|Block|Nagar|Colony)\b[^,\n]*",
            re.IGNORECASE,
        ),
        "[MASKED_ADDRESS]",
    ),
]


def mask_personal_data(text: str) -> tuple[str, list[str]]:
    """Mask phone numbers, addresses, emails, IDs, and card numbers in text."""
    if not text:
        return "", []
    masked = text
    redacted_types: list[str] = []
    for label, pattern, replacement in PII_RULES:
        if pattern.search(masked):
            # Avoid masking customer/ticket IDs like CUST-1001, TKT-5521, CALL-8821
            def _safe_sub(match: re.Match[str]) -> str:
                token = match.group(0)
                if len(token.strip()) < 7:
                    return token
                return replacement

            new_masked = pattern.sub(_safe_sub, masked)
            if new_masked != masked:
                redacted_types.append(label)
                masked = new_masked
    return masked, redacted_types
