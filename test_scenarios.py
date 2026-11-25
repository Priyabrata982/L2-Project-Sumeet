"""Golden test scenarios T1-T8 from the Case Study specification (Pages 8-9)."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_assistant.graph import ChurnInsightService
from churn_assistant.schemas import DRIVER_ORDER


def _svc() -> ChurnInsightService:
    return ChurnInsightService()


def test_t1_happy_path_cust_1001():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1001",
            "question": "Why might this customer leave, and what should I do?",
            "context": {
                "channel": "inbound_call",
                "stated_reason": "calls keep dropping at home",
            },
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "COMPLETED"
    out = res["response"]
    assert out["risk"]["level"] == "high"
    assert out["risk"]["score"] == 6
    driver_names = [d["driver"] for d in out["drivers"]]
    assert driver_names == ["network_quality", "usage_decline"]
    assert out["recommended_action"]["action"] == "network_escalation"
    assert out["escalate"] is False
    # Every citation resolves to verified records/policies
    all_cites = [c for d in out["drivers"] for c in d["citations"]] + out["recommended_action"]["citations"]
    assert "TKT-5521" in all_cites
    assert "TKT-5540" in all_cites
    assert "TKT-5562" in all_cites
    assert "USG-1001@2026-09" in all_cites
    assert "NET-ESC@v1" in all_cites


def test_t2_unknown_customer_cust_9999():
    res = _svc().analyze(
        {
            "customer_id": "CUST-9999",
            "question": "Why might this customer leave?",
            "context": {"channel": "inbound_call", "stated_reason": "billing"},
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "COMPLETED"
    out = res["response"]
    assert out["drivers"] == []
    assert out["risk"] is None
    assert out["recommended_action"] is None
    assert any("CUST-9999" in item for item in out["unsupported"])


def test_t3_unverified_content_note_77_never_cited():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1001",
            "question": "Why might this customer leave, and what should I do?",
            "context": {
                "channel": "inbound_call",
                "stated_reason": "calls keep dropping at home",
            },
            "session_date": "2026-10-15",
        }
    )
    out = res["response"]
    all_cites = [c for d in out["drivers"] for c in d["citations"]] + out["recommended_action"]["citations"]
    assert "NOTE-77" not in all_cites
    assert out["risk"]["score"] == 6
    assert "competitor_intent" not in [d["driver"] for d in out["drivers"]]


def test_t4_conflicting_data_cust_1002():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1002",
            "question": "What is the status of the roaming double charge?",
            "context": {"channel": "inbound_call", "stated_reason": "billing complaint"},
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "COMPLETED"
    out = res["response"]
    conflicts_text = " ".join(out["data_conflicts"])
    assert "TKT-6610" in conflicts_text
    assert "CALL-8821" in conflicts_text
    # Latest verified record (CALL-8821) wins -> open_disputed_billing -> bill_correction
    assert "CALL-8821" in out["drivers"][0]["citations"]
    assert out["recommended_action"]["action"] == "bill_correction"


def test_t5_unsupported_competitor_pricing_question():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1004",
            "question": "Is the competitor's plan cheaper for CUST-1004?",
            "context": {"channel": "inbound_call", "stated_reason": "asking competitor pricing"},
            "session_date": "2026-10-15",
        }
    )
    out = res["response"]
    assert any("competitor pricing" in u.lower() for u in out["unsupported"])
    assert "cheaper" not in out["summary"].lower()


def test_t6_expired_offer_cust_1003_fixed_session_date():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1003",
            "question": "Can I give CUST-1003 the festive offer?",
            "context": {"channel": "inbound_call", "stated_reason": "wants festive discount"},
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "COMPLETED"
    out = res["response"]
    assert any("OFFER-FEST" in u and "expired" in u.lower() for u in out["unsupported"])
    assert out["recommended_action"] is None
    assert "50%" not in out["summary"]
    assert "half rental" not in out["summary"].lower()


def test_t7_policy_exception_cust_1004_escalates_and_pauses():
    svc = _svc()
    res = svc.analyze(
        {
            "customer_id": "CUST-1004",
            "question": "Customer wants 50% off for 6 months",
            "context": {"channel": "inbound_call", "stated_reason": "port-out threat"},
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "ESCALATED_PENDING_SUPERVISOR"
    assert res["response"]["escalate"] is True
    assert res["response"]["escalation_reason"]
    assert res["interrupt"]["interrupt_type"] == "supervisor_review"

    # Resume at Supervisor review node
    final = svc.resolve_supervisor_review(
        res["thread_id"], decision="reject", reviewer="Lead Supervisor", comments="Offer 20% for 3m max."
    )
    assert final["status"] == "COMPLETED"
    assert "Lead Supervisor" in final["response"]["summary"]


def test_t8_sparse_history_cust_1005():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1005",
            "question": "",
            "context": {"channel": "inbound_call", "stated_reason": ""},
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "COMPLETED"
    out = res["response"]
    assert out["drivers"] == []
    assert out["risk"] is None
    assert out["recommended_action"] is None
    missing_text = " ".join(out["missing_information"]).lower()
    assert "interactions" in missing_text
    assert "usage" in missing_text


def test_extended_customer_cust_1006_driver_order():
    res = _svc().analyze(
        {
            "customer_id": "CUST-1006",
            "question": "Why is customer asking for plan change?",
            "context": {"channel": "inbound_call", "stated_reason": "wants cheaper plan"},
            "session_date": "2026-10-15",
        }
    )
    assert res["status"] == "COMPLETED"
    out = res["response"]
    assert out["risk"]["level"] == "medium"
    assert out["risk"]["score"] == 3  # usage_drop_30pct (+2) + repeat_contact (+1)
    assert out["recommended_action"]["action"] == "plan_review"
    order_pos = {name: i for i, name in enumerate(DRIVER_ORDER)}
    positions = [order_pos[d["driver"]] for d in out["drivers"]]
    assert positions == sorted(positions)
