"""Tests for PII masking, guard invariants, and FastAPI POST /insight."""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from api import app
from churn_assistant.guards import run_guard_checks
from churn_assistant.pii import mask_personal_data
from churn_assistant.schemas import DriverItem, InsightResponse, RecommendedAction, RiskAssessment


def test_pii_masking_removes_phone_email_id_card():
    text = (
        "Call me at +91 9876543210 or john.doe@example.com; "
        "PAN ABCDE1234F, card 4111-2222-3333-4444, 42 MG Road."
    )
    masked, redacted = mask_personal_data(text)
    assert "9876543210" not in masked
    assert "john.doe@example.com" not in masked
    assert "ABCDE1234F" not in masked
    assert "4111-2222-3333-4444" not in masked
    assert "42 MG Road" not in masked
    assert "phone" in redacted
    assert "email" in redacted
    assert "pan_id" in redacted
    assert "card_number" in redacted
    assert "address" in redacted


def test_guard_blocks_unverified_citation_and_enforces_code_risk():
    # Simulate an LLM draft trying to cite NOTE-77 (verified: false) and downgrade risk to low
    bad_draft = InsightResponse(
        customer_id="CUST-1001",
        summary="Draft output",
        drivers=[
            DriverItem(driver="competitor_intent", evidence="Draft note", citations=["NOTE-77"]),
            DriverItem(driver="network_quality", evidence="3 drops", citations=["TKT-5521"]),
        ],
        risk=RiskAssessment(level="low", score=1, signals_used=["repeat_contact"]),
        recommended_action=RecommendedAction(
            action="network_escalation",
            reasoning="Threshold met",
            based_on_signals=["network_complaints_3plus"],
            citations=["TKT-5562", "NET-ESC@v1"],
        ),
    )
    computed_risk = RiskAssessment(
        level="high",
        score=6,
        signals_used=["network_complaints_3plus", "usage_drop_30pct", "repeat_contact"],
    )
    checked = run_guard_checks(
        draft=bad_draft,
        known_customer=True,
        question="Why might customer leave?",
        stated_reason="dropped calls",
        verified_records=[
            {"record_id": "TKT-5521"},
            {"record_id": "TKT-5562"},
        ],
        unverified_record_ids=["NOTE-77"],
        active_policies=[{"doc_id": "NET-ESC", "citation_id": "NET-ESC@v1"}],
        expired_policies=[],
        data_conflicts=[],
        has_unresolved_conflict=False,
        computed_risk=computed_risk,
    )
    # NOTE-77 driver stripped; code risk enforced; escalated due to citation failure
    all_cites = [c for d in checked.drivers for c in d.citations]
    assert "NOTE-77" not in all_cites
    assert checked.risk.level == "high"
    assert checked.risk.score == 6
    assert checked.escalate is True
    assert checked.escalation_reason is not None


def test_guard_flags_unverified_citation_in_recommended_action():
    draft = InsightResponse(
        customer_id="CUST-1001",
        summary="Draft output",
        drivers=[
            DriverItem(driver="network_quality", evidence="3 drops", citations=["TKT-5521"]),
        ],
        risk=RiskAssessment(level="high", score=6, signals_used=["network_complaints_3plus"]),
        recommended_action=RecommendedAction(
            action="network_escalation",
            reasoning="Threshold met",
            based_on_signals=["network_complaints_3plus"],
            citations=["NOTE-77", "NET-ESC@v1"],
        ),
    )
    computed_risk = RiskAssessment(
        level="high",
        score=6,
        signals_used=["network_complaints_3plus"],
    )
    checked = run_guard_checks(
        draft=draft,
        known_customer=True,
        question="Why might customer leave?",
        stated_reason="dropped calls",
        verified_records=[{"record_id": "TKT-5521"}],
        unverified_record_ids=["NOTE-77"],
        active_policies=[{"doc_id": "NET-ESC", "citation_id": "NET-ESC@v1"}],
        expired_policies=[],
        data_conflicts=[],
        has_unresolved_conflict=False,
        computed_risk=computed_risk,
    )
    assert "NOTE-77" not in checked.recommended_action.citations
    assert checked.escalate is True


def test_fastapi_post_insight():
    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/insight",
                json={
                    "customer_id": "CUST-1001",
                    "question": "Why might this customer leave, and what should I do?",
                    "context": {
                        "channel": "inbound_call",
                        "stated_reason": "calls keep dropping at home",
                    },
                    "session_date": "2026-10-15",
                },
            )

    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        loop = asyncio.SelectorEventLoop()
        try:
            resp = loop.run_until_complete(_call())
        finally:
            loop.close()
    else:
        resp = asyncio.run(_call())

    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == "CUST-1001"
    assert data["risk"]["level"] == "high"
    assert data["recommended_action"]["action"] == "network_escalation"
