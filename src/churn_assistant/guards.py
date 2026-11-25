"""Code-enforced guard checks for citations, policy limits, conflicts, and escalation."""

from __future__ import annotations

import re
from typing import Any, Optional

from .schemas import (
    ACTION_CATALOG,
    DRIVER_ORDER,
    DriverItem,
    InsightResponse,
    RecommendedAction,
    RiskAssessment,
)
from .scoring import risk_band_from_score

DISCOUNT_PCT_RE = re.compile(r"(\d{1,3})\s*%")
DURATION_MONTHS_RE = re.compile(r"(\d{1,2})\s*months?", re.IGNORECASE)
CREDIT_RS_RE = re.compile(r"rs\.?\s*(\d{2,6})", re.IGNORECASE)


def _detect_policy_exception_in_request(
    question: str,
    stated_reason: str,
    verified_records: list[dict[str, Any]],
    active_policies: list[dict[str, Any]],
) -> Optional[str]:
    """Detect whether customer or agent request exceeds RET-POL authority limits."""
    ret_pol = next((p for p in active_policies if p.get("doc_id") == "RET-POL"), {})
    max_pct = int(ret_pol.get("max_discount_pct", 20))
    max_months = int(ret_pol.get("max_discount_months", 3))
    max_credit = int(ret_pol.get("max_bill_credit_rs", 500))

    reasons: list[str] = []

    # Check structured fields in verified records (e.g. CALL-9034: 50% off for 6 months)
    for rec in verified_records:
        req_pct = rec.get("requested_discount_pct")
        req_months = rec.get("requested_duration_months")
        req_credit = rec.get("requested_credit_rs")
        if req_pct is not None and int(req_pct) > max_pct:
            reasons.append(
                f"Record {rec['record_id']} requests {req_pct}% discount (RET-POL limit: {max_pct}%)"
            )
        if req_months is not None and int(req_months) > max_months:
            reasons.append(
                f"Record {rec['record_id']} requests {req_months} months duration (RET-POL limit: {max_months} months)"
            )
        if req_credit is not None and int(req_credit) > max_credit:
            reasons.append(
                f"Record {rec['record_id']} requests Rs {req_credit} bill credit (RET-POL limit: Rs {max_credit})"
            )

    # Also check agent question / stated_reason text for over-limit asks
    combined_text = f"{question} {stated_reason}"
    for m in DISCOUNT_PCT_RE.finditer(combined_text):
        val = int(m.group(1))
        if val > max_pct and val <= 100:
            reasons.append(f"Requested {val}% discount exceeds RET-POL max {max_pct}%")
    for m in DURATION_MONTHS_RE.finditer(combined_text):
        val = int(m.group(1))
        if val > max_months and ("off" in combined_text.lower() or "discount" in combined_text.lower()):
            reasons.append(f"Requested {val}-month discount exceeds RET-POL max {max_months} months")
    for m in CREDIT_RS_RE.finditer(combined_text):
        val = int(m.group(1))
        if val > max_credit and "credit" in combined_text.lower():
            reasons.append(f"Requested Rs {val} credit exceeds RET-POL max Rs {max_credit}")

    if reasons:
        deduped = list(dict.fromkeys(reasons))
        return "; ".join(deduped) + ". Supervisor approval required."
    return None


def run_guard_checks(
    draft: InsightResponse,
    known_customer: bool,
    question: str,
    stated_reason: str,
    verified_records: list[dict[str, Any]],
    unverified_record_ids: list[str],
    active_policies: list[dict[str, Any]],
    expired_policies: list[dict[str, Any]],
    data_conflicts: list[str],
    has_unresolved_conflict: bool,
    computed_risk: Optional[RiskAssessment],
) -> InsightResponse:
    """Enforce all Page 7 invariants and determine escalation in code."""
    valid_record_ids = {r["record_id"] for r in verified_records}
    valid_policy_ids = set()
    for p in active_policies:
        valid_policy_ids.add(p["doc_id"])
        valid_policy_ids.add(p["citation_id"])

    allowed_citations = valid_record_ids | valid_policy_ids
    unverified_set = set(unverified_record_ids)

    escalate = False
    escalation_reasons: list[str] = []

    # 1. Unknown customer short-circuit
    if not known_customer:
        unsupported = list(draft.unsupported)
        if not any(draft.customer_id in u for u in unsupported):
            unsupported.append(f"Unknown customer ID: {draft.customer_id}")
        missing = list(draft.missing_information) or [
            f"Verified customer profile for {draft.customer_id}"
        ]
        return InsightResponse(
            customer_id=draft.customer_id,
            summary=f"Unknown customer ID {draft.customer_id}; turned back at input check.",
            drivers=[],
            risk=None,
            recommended_action=None,
            missing_information=missing,
            unsupported=unsupported,
            data_conflicts=[],
            escalate=False,
            escalation_reason=None,
        )

    # 2. Enforce citations resolve and never reference verified: false records
    cleaned_drivers: list[DriverItem] = []
    citation_failure = False
    for drv in draft.drivers:
        if drv.driver not in DRIVER_ORDER:
            citation_failure = True
            continue
        bad_unverified = [c for c in drv.citations if c in unverified_set]
        if bad_unverified:
            citation_failure = True
        valid_cites = [c for c in drv.citations if c in valid_record_ids]
        if not valid_cites:
            citation_failure = True
            continue
        cleaned_drivers.append(
            DriverItem(driver=drv.driver, evidence=drv.evidence, citations=valid_cites)
        )

    order_idx = {name: i for i, name in enumerate(DRIVER_ORDER)}
    cleaned_drivers.sort(key=lambda d: order_idx[d.driver])

    # 3. Enforce code-computed risk band (model cannot override)
    final_risk = computed_risk
    missing_info = list(draft.missing_information)
    if final_risk is not None:
        expected_level = risk_band_from_score(final_risk.score)
        if final_risk.level != expected_level:
            final_risk = RiskAssessment(
                level=expected_level,
                score=final_risk.score,
                signals_used=final_risk.signals_used,
            )
    else:
        # Rule: risk may be null only when no signals exist, and missing_information must not be empty
        if not missing_info:
            missing_info = [
                "recent customer interactions (tickets, calls, billing disputes)",
                "usage trend metrics",
            ]

    # 4. Enforce recommended_action catalog, based_on_signals, and citations
    final_action: Optional[RecommendedAction] = None
    if draft.recommended_action is not None:
        act = draft.recommended_action
        scored_signals = set(final_risk.signals_used) if final_risk else set()
        valid_based_on = [s for s in act.based_on_signals if s in scored_signals]
        bad_act_cites = [
            c for c in act.citations if c in unverified_set or c not in allowed_citations
        ]
        if bad_act_cites:
            citation_failure = True
        valid_act_cites = [c for c in act.citations if c in allowed_citations]
        if (
            act.action in ACTION_CATALOG
            and len(valid_based_on) >= 1
            and len(valid_act_cites) >= 1
        ):
            # Ensure no expired offer text or over-limit discount appears in reasoning
            expired_ids = [p["doc_id"] for p in expired_policies]
            reasoning_text = act.reasoning
            for exp_id in expired_ids:
                if exp_id.lower() in reasoning_text.lower():
                    reasoning_text = (
                        f"{act.action.replace('_', ' ').title()} recommended within active policy; "
                        f"expired offer {exp_id} excluded."
                    )
            final_action = RecommendedAction(
                action=act.action,
                reasoning=reasoning_text,
                based_on_signals=valid_based_on,
                citations=valid_act_cites,
                follow_up=act.follow_up,
            )
        elif final_risk and final_risk.signals_used:
            citation_failure = True

    if citation_failure:
        escalate = True
        escalation_reasons.append(
            "Guard check detected unverified or unresolved citation in draft output."
        )

    # 5. Enforce RET-POL offer limits & detect policy exceptions
    exc_reason = _detect_policy_exception_in_request(
        question=question,
        stated_reason=stated_reason,
        verified_records=verified_records,
        active_policies=active_policies,
    )
    if exc_reason:
        escalate = True
        escalation_reasons.append(exc_reason)

    # 6. Unresolved conflicts escalate
    if has_unresolved_conflict:
        escalate = True
        escalation_reasons.append(
            "Unresolved data conflict across same-date verified records requires supervisor review."
        )

    final_escalation_reason = "; ".join(escalation_reasons) if escalate else None

    return InsightResponse(
        customer_id=draft.customer_id,
        summary=draft.summary,
        drivers=cleaned_drivers,
        risk=final_risk,
        recommended_action=final_action,
        missing_information=missing_info,
        unsupported=list(dict.fromkeys(draft.unsupported)),
        data_conflicts=list(dict.fromkeys(data_conflicts)),
        escalate=escalate,
        escalation_reason=final_escalation_reason,
    )
