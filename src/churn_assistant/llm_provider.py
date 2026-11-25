"""LLM generation (Ollama gpt-oss:120b-cloud) with deterministic grounded fallback."""

from __future__ import annotations

import json
import socket
import urllib.parse
import urllib.request
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .schemas import (
    ACTION_CATALOG,
    DRIVER_ORDER,
    DriverItem,
    InsightResponse,
    RecommendedAction,
    RiskAssessment,
)

SYSTEM_PROMPT = """You are a Telecom Customer Churn Insight Assistant.
Reason ONLY from the verified customer records and active retention policies provided.

STRICT RULES:
1. NEVER change or override the code-computed risk.score, risk.level, or risk.signals_used.
2. Separate signals (cited to an exact record_id in verified_records) from recommendations.
3. NEVER cite any unverified record or invent customer history.
4. Use ONLY driver names from: network_quality, billing_dispute, plan_fit, service_experience, competitor_intent, usage_decline (in that order).
5. Use ONLY recommended_action.action from: plan_review, network_escalation, bill_correction, callback.
6. NEVER compare competitor prices or quote network repair dates (not in approved data).
7. NEVER suggest an expired offer or an offer exceeding RET-POL limits (max Rs 500 bill credit, max 20% discount for 3 months).
8. Return ONLY valid JSON matching InsightResponse.
"""


def is_ollama_available() -> bool:
    """Fast check whether Ollama server is reachable."""
    if settings.use_fallback_llm:
        return False
    try:
        parsed = urllib.parse.urlparse(settings.ollama_base_url)
        host = parsed.hostname or "127.0.0.1"
        if host == "localhost":
            host = "127.0.0.1"
        port = parsed.port or 11434
        with socket.create_connection((host, port), timeout=0.3):
            pass
        req = urllib.request.Request(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False


def _deterministic_draft(
    customer_id: str,
    known_customer: bool,
    question: str,
    stated_reason: str,
    session_date: str,
    verified_records: list[dict[str, Any]],
    active_policies: list[dict[str, Any]],
    expired_policies: list[dict[str, Any]],
    data_conflicts: list[str],
    risk: Optional[RiskAssessment],
) -> InsightResponse:
    """Deterministic grounded insight builder used when Ollama is offline or as baseline."""
    q_lower = f"{question} {stated_reason}".lower()
    unsupported: list[str] = []
    missing_info: list[str] = []

    if not known_customer:
        return InsightResponse(
            customer_id=customer_id,
            summary=f"Unknown customer ID {customer_id}; request turned back at input check with no records retrieved.",
            drivers=[],
            risk=None,
            recommended_action=None,
            missing_information=[f"Verified customer profile for {customer_id}"],
            unsupported=[f"Unknown customer ID: {customer_id}"],
            data_conflicts=[],
            escalate=False,
            escalation_reason=None,
        )

    # Check unsupported queries
    if any(kw in q_lower for kw in ("competitor", "cheaper", "price comparison", "pricing")):
        unsupported.append(
            "Competitor pricing is not in approved content; no price comparison can be given."
        )
    if any(kw in q_lower for kw in ("repair date", "when will", "fixed", "eta")):
        unsupported.append("Network repair dates are not in approved content.")

    # Check expired offer mentions
    for exp in expired_policies:
        if exp["doc_id"].lower() in q_lower or "festive" in q_lower or "half rental" in q_lower:
            unsupported.append(
                f"Expired offer {exp['doc_id']} (valid {exp['valid_from']} to {exp['valid_to']}, "
                f"expired before session date {session_date}) cannot be recommended or quoted."
            )

    # Sparse history case (no verified interaction or usage records)
    if not verified_records:
        return InsightResponse(
            customer_id=customer_id,
            summary=(
                f"Customer {customer_id} has plan metadata only and no verified interactions "
                "or usage trend records in the last 90 days."
            ),
            drivers=[],
            risk=None,
            recommended_action=None,
            missing_information=[
                "recent interactions (tickets, calls, billing disputes)",
                "usage trend metrics",
            ],
            unsupported=unsupported,
            data_conflicts=data_conflicts,
            escalate=False,
            escalation_reason=None,
        )

    # Build drivers strictly in DRIVER_ORDER
    drivers: list[DriverItem] = []

    net_recs = [r for r in verified_records if r.get("category") == "network_quality"]
    if net_recs:
        cites = [r["record_id"] for r in net_recs]
        if len(net_recs) >= 3:
            evidence = f"{len(net_recs)} dropped-call tickets from the same cell in 30 days"
        else:
            evidence = net_recs[-1]["summary"]
        drivers.append(
            DriverItem(driver="network_quality", evidence=evidence, citations=cites)
        )

    bill_recs = [r for r in verified_records if r.get("category") == "billing_dispute"]
    if bill_recs:
        latest_bill = bill_recs[-1]
        drivers.append(
            DriverItem(
                driver="billing_dispute",
                evidence=latest_bill["summary"],
                citations=[latest_bill["record_id"]],
            )
        )

    plan_recs = [r for r in verified_records if r.get("category") == "plan_fit"]
    if plan_recs:
        drivers.append(
            DriverItem(
                driver="plan_fit",
                evidence=plan_recs[-1]["summary"],
                citations=[r["record_id"] for r in plan_recs],
            )
        )

    svc_recs = [r for r in verified_records if r.get("category") == "service_experience"]
    if svc_recs:
        drivers.append(
            DriverItem(
                driver="service_experience",
                evidence=svc_recs[-1]["summary"],
                citations=[r["record_id"] for r in svc_recs],
            )
        )

    comp_recs = [r for r in verified_records if r.get("category") == "competitor_intent"]
    if comp_recs:
        drivers.append(
            DriverItem(
                driver="competitor_intent",
                evidence=comp_recs[-1]["summary"],
                citations=[r["record_id"] for r in comp_recs],
            )
        )

    usage_recs = [
        r
        for r in verified_records
        if r.get("record_type") == "usage_metric" and float(r.get("usage_change_pct", 0)) <= -30
    ]
    if usage_recs:
        pct = abs(int(usage_recs[-1]["usage_change_pct"]))
        drivers.append(
            DriverItem(
                driver="usage_decline",
                evidence=f"Data usage down {pct}% month on month",
                citations=[r["record_id"] for r in usage_recs],
            )
        )

    # Sort drivers explicitly by DRIVER_ORDER
    order_idx = {name: i for i, name in enumerate(DRIVER_ORDER)}
    drivers.sort(key=lambda d: order_idx[d.driver])

    # Choose one retention action from fixed catalog when scored signals exist
    signals = risk.signals_used if risk else []
    recommended_action: Optional[RecommendedAction] = None

    if "network_complaints_3plus" in signals:
        latest_net = net_recs[-1]["record_id"]
        recommended_action = RecommendedAction(
            action="network_escalation",
            reasoning="Meets the NET-ESC threshold (3+ complaints from one location in 30 days); network quality is the leading driver.",
            based_on_signals=["network_complaints_3plus"],
            citations=[latest_net, "NET-ESC@v1"],
            follow_up="callback",
        )
        missing_info.append("customer's preferred callback time")
        if "Network repair dates are not in approved content." not in unsupported:
            unsupported.append("Network repair dates are not in approved content.")
    elif "open_disputed_billing" in signals:
        latest_bill_id = bill_recs[-1]["record_id"]
        recommended_action = RecommendedAction(
            action="bill_correction",
            reasoning=(
                f"Latest verified record ({latest_bill_id}) confirms billing dispute remains open; "
                "BILL-POL requires correction after verification within 48 hours."
            ),
            based_on_signals=["open_disputed_billing"],
            citations=[latest_bill_id, "BILL-POL@v1"],
            follow_up="callback",
        )
        missing_info.append("billing verification confirmation ID")
    elif "competitor_or_port_out" in signals:
        latest_comp_id = comp_recs[-1]["record_id"]
        recommended_action = RecommendedAction(
            action="plan_review",
            reasoning=(
                "Port-out intent and competitor mention detected; within-policy plan review "
                "(up to 20% discount for 3 months per RET-POL) requires supervisor review for larger requests."
            ),
            based_on_signals=["competitor_or_port_out"],
            citations=[latest_comp_id, "RET-POL@v1"],
            follow_up="callback",
        )
    elif "usage_drop_30pct" in signals or "repeat_contact" in signals:
        primary_sig = "usage_drop_30pct" if "usage_drop_30pct" in signals else "repeat_contact"
        cite_id = (plan_recs[-1]["record_id"] if plan_recs else verified_records[-1]["record_id"])
        recommended_action = RecommendedAction(
            action="plan_review",
            reasoning="Usage decline and repeat plan inquiries indicate plan mismatch; review within RET-POL limits.",
            based_on_signals=[primary_sig],
            citations=[cite_id, "RET-POL@v1"],
            follow_up="callback",
        )

    # Concise summary
    if drivers:
        driver_summary = "; ".join(f"{d.driver}: {d.evidence}" for d in drivers)
        summary = f"Customer {customer_id} verified signals — {driver_summary}."
    else:
        summary = f"Customer {customer_id} has verified low-risk records with no active churn signals."

    if any(exp["doc_id"] == "OFFER-FEST" for exp in expired_policies) and (
        "festive" in q_lower or "offer" in q_lower
    ):
        summary += " Note: OFFER-FEST is expired and cannot be applied."

    return InsightResponse(
        customer_id=customer_id,
        summary=summary,
        drivers=drivers,
        risk=risk,
        recommended_action=recommended_action,
        missing_information=missing_info,
        unsupported=unsupported,
        data_conflicts=data_conflicts,
        escalate=False,
        escalation_reason=None,
    )


def generate_insight_draft(
    customer_id: str,
    known_customer: bool,
    customer_profile: dict[str, Any],
    question: str,
    stated_reason: str,
    session_date: str,
    verified_records: list[dict[str, Any]],
    active_policies: list[dict[str, Any]],
    expired_policies: list[dict[str, Any]],
    data_conflicts: list[str],
    risk: Optional[RiskAssessment],
) -> tuple[InsightResponse, str]:
    """Generate structured insight using Ollama when available, else deterministic builder."""
    baseline = _deterministic_draft(
        customer_id=customer_id,
        known_customer=known_customer,
        question=question,
        stated_reason=stated_reason,
        session_date=session_date,
        verified_records=verified_records,
        active_policies=active_policies,
        expired_policies=expired_policies,
        data_conflicts=data_conflicts,
        risk=risk,
    )

    # Short-circuit unknown customer or sparse history without invoking model
    if not known_customer or not verified_records:
        return baseline, "deterministic_rule"

    if is_ollama_available():
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.0,
            )
            structured_llm = llm.with_structured_output(InsightResponse)

            prompt_payload = {
                "customer_id": customer_id,
                "customer_profile": customer_profile,
                "session_date": session_date,
                "agent_question": question,
                "stated_reason": stated_reason,
                "fixed_risk": risk.model_dump() if risk else None,
                "verified_records": verified_records,
                "active_policies": active_policies,
                "expired_policies": expired_policies,
                "data_conflicts": data_conflicts,
                "allowed_drivers_order": list(DRIVER_ORDER),
                "allowed_action_catalog": list(ACTION_CATALOG),
            }
            result = structured_llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(prompt_payload, indent=2)),
                ]
            )
            if isinstance(result, InsightResponse):
                return result, f"ollama:{settings.ollama_model}"
        except Exception:
            pass

    return baseline, "deterministic_fallback"
