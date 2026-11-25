"""Fixed deterministic signal-scoring rules for churn risk classification."""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any, Optional

from .schemas import RiskAssessment

SIGNAL_WEIGHTS: dict[str, int] = {
    "network_complaints_3plus": 3,
    "open_disputed_billing": 3,
    "competitor_or_port_out": 3,
    "usage_drop_30pct": 2,
    "repeat_contact": 1,
}


def risk_band_from_score(score: int) -> str:
    """Map fixed numeric score to fixed risk band: 0-2 low, 3-5 medium, 6+ high."""
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    return "high"


def compute_churn_risk(
    verified_records: list[dict[str, Any]], session_date: str
) -> Optional[RiskAssessment]:
    """Compute churn risk strictly from verified records using fixed weights.

    Returns None when the customer has no verified interaction or usage records at all
    (sparse history case).
    """
    if not verified_records:
        return None

    sess_dt = date.fromisoformat(session_date)
    signals_used: list[str] = []

    # 1. Network complaints >= 3 from same location within 30 days (+3)
    recent_network = []
    for rec in verified_records:
        if rec.get("category") == "network_quality" and rec.get("record_type") in (
            "ticket",
            "call_transcript",
        ):
            rec_dt = date.fromisoformat(rec["date"])
            if 0 <= (sess_dt - rec_dt).days <= 30:
                recent_network.append(rec)

    cell_counts = Counter(r.get("cell_id", "unknown") for r in recent_network)
    if any(count >= 3 for count in cell_counts.values()):
        signals_used.append("network_complaints_3plus")

    # 2. Open or disputed billing issue (+3), using latest record per dispute_group
    billing_groups: dict[str, dict[str, Any]] = {}
    standalone_billing: list[dict[str, Any]] = []
    for rec in sorted(verified_records, key=lambda r: r.get("date", "")):
        if rec.get("category") == "billing_dispute":
            grp = rec.get("dispute_group")
            if grp:
                billing_groups[grp] = rec  # Latest verified record wins
            else:
                standalone_billing.append(rec)

    effective_billing = list(billing_groups.values()) + standalone_billing
    if any(r.get("status") in ("open", "open_dispute") for r in effective_billing):
        signals_used.append("open_disputed_billing")

    # 3. Competitor mention or port-out request (+3)
    if any(
        rec.get("category") == "competitor_intent" or rec.get("port_out_request") is True
        for rec in verified_records
    ):
        signals_used.append("competitor_or_port_out")

    # 4. Usage down more than 30% month on month (+2)
    if any(
        rec.get("record_type") == "usage_metric"
        and float(rec.get("usage_change_pct", 0)) <= -30.0
        for rec in verified_records
    ):
        signals_used.append("usage_drop_30pct")

    # 5. Repeat contact for the same issue (+1)
    # Count interactions (tickets / call transcripts) per category or dispute_group
    issue_counts: Counter[str] = Counter()
    for rec in verified_records:
        if rec.get("record_type") in ("ticket", "call_transcript"):
            key = rec.get("dispute_group") or rec.get("cell_id") or rec.get("category", "general")
            issue_counts[key] += 1
    if any(count >= 2 for count in issue_counts.values()):
        signals_used.append("repeat_contact")

    score = sum(SIGNAL_WEIGHTS[s] for s in signals_used)
    return RiskAssessment(
        level=risk_band_from_score(score),
        score=score,
        signals_used=signals_used,
    )
