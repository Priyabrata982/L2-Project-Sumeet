"""Streamlit UI for the Telecom Customer Churn Insight Assistant (Page 8 layout)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_assistant.config import settings
from churn_assistant.graph import ChurnInsightService
from churn_assistant.llm_provider import is_ollama_available
from churn_assistant.scoring import SIGNAL_WEIGHTS

st.set_page_config(
    page_title="Telecom Churn Insight Assistant",
    page_icon="📡",
    layout="wide",
)


def get_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "T1 · Happy path (CUST-1001)": {
            "customer_id": "CUST-1001",
            "question": "Why might this customer leave, and what should I do?",
            "channel": "inbound_call",
            "stated_reason": "calls keep dropping at home",
            "session_date": "2026-10-15",
        },
        "T2 · Unknown customer (CUST-9999)": {
            "customer_id": "CUST-9999",
            "question": "Why might this customer leave?",
            "channel": "inbound_call",
            "stated_reason": "billing inquiry",
            "session_date": "2026-10-15",
        },
        "T4 · Conflicting data (CUST-1002)": {
            "customer_id": "CUST-1002",
            "question": "What happened with the roaming charge?",
            "channel": "inbound_call",
            "stated_reason": "billing complaint",
            "session_date": "2026-10-15",
        },
        "T5 · Unsupported question (CUST-1004)": {
            "customer_id": "CUST-1004",
            "question": "Is the competitor's plan cheaper for CUST-1004?",
            "channel": "inbound_call",
            "stated_reason": "asking about competitor plan",
            "session_date": "2026-10-15",
        },
        "T6 · Expired offer (CUST-1003)": {
            "customer_id": "CUST-1003",
            "question": "Can I give CUST-1003 the festive offer?",
            "channel": "inbound_call",
            "stated_reason": "asking for discount",
            "session_date": "2026-10-15",
        },
        "T7 · Policy exception (CUST-1004)": {
            "customer_id": "CUST-1004",
            "question": "Customer wants 50% off for 6 months or will port out.",
            "channel": "inbound_call",
            "stated_reason": "CUST-1004 wants 50% off for 6 months",
            "session_date": "2026-10-15",
        },
        "T8 · Sparse history (CUST-1005)": {
            "customer_id": "CUST-1005",
            "question": "",
            "channel": "inbound_call",
            "stated_reason": "",
            "session_date": "2026-10-15",
        },
        "Custom · Extended Customer (CUST-1006)": {
            "customer_id": "CUST-1006",
            "question": "Why is data usage dropping and what action is within policy?",
            "channel": "inbound_call",
            "stated_reason": "bill feels too high",
            "session_date": "2026-10-15",
        },
    }


@st.cache_resource
def get_service() -> ChurnInsightService:
    return ChurnInsightService()


service = get_service()

if "churn_result" not in st.session_state:
    st.session_state.churn_result = None
if "scenario_key" not in st.session_state:
    st.session_state.scenario_key = list(get_scenarios().keys())[0]

with st.sidebar:
    st.markdown("### Golden Scenarios")
    scenarios = get_scenarios()
    chosen = st.selectbox("Load test scenario", list(scenarios.keys()))
    if st.button("Load Scenario", use_container_width=True):
        st.session_state.scenario_key = chosen
        st.session_state.churn_result = None
        st.rerun()

    st.divider()
    ollama_up = is_ollama_available()
    st.caption(
        f"LLM: {'🟢 Ollama (' + settings.ollama_model + ')' if ollama_up else '🟡 Grounded Fallback (Ollama offline)'}"
    )

preset = scenarios[st.session_state.scenario_key]

st.title("Telecom Churn Insight Assistant")
left_col, right_col = st.columns([1.0, 1.25], gap="large")

with left_col:
    with st.form("insight_request_form"):
        customer_id = st.text_input("Customer ID", value=preset["customer_id"])
        question = st.text_area("Agent question", value=preset["question"], height=80)
        channel = st.selectbox(
            "Channel",
            ["inbound_call", "chat", "email", "retail_store"],
            index=0,
        )
        stated_reason = st.text_input(
            "Customer's stated reason", value=preset["stated_reason"]
        )
        session_date = st.text_input(
            "Session date (YYYY-MM-DD)", value=preset["session_date"]
        )
        analyse = st.form_submit_button(
            "Analyse", type="primary", use_container_width=True
        )

    if analyse:
        req = {
            "customer_id": customer_id.strip(),
            "question": question,
            "context": {
                "channel": channel,
                "stated_reason": stated_reason,
            },
            "session_date": session_date.strip() or settings.default_session_date,
        }
        st.session_state.churn_result = service.analyze(req)
        st.rerun()

with right_col:
    res = st.session_state.churn_result
    if not res:
        st.info("Enter a customer ID and click **Analyse** to view risk and recommended action.")
    else:
        resp = res["response"]
        risk = resp.get("risk")
        action = resp.get("recommended_action")

        # 1. Churn risk FIRST (Page 8: readable in under 10 seconds)
        st.markdown("**Churn risk**")
        if risk is None:
            st.warning("Null · no verified interaction signals")
        else:
            lvl = risk["level"]
            score = risk["score"]
            badge_text = f"**{lvl.title()} · score {score}**"
            if lvl == "high":
                st.error(badge_text)
            elif lvl == "medium":
                st.warning(badge_text)
            else:
                st.success(badge_text)

        # 2. Recommended action SECOND
        st.markdown("**Recommended action**")
        if action:
            follow = f"; {action['follow_up']} to follow" if action.get("follow_up") else ""
            st.success(
                f"**{action['action'].replace('_', ' ').title()}**\n\n"
                f"{action['reasoning']} *(citations: {', '.join(action['citations'])}{follow})*"
            )
        else:
            st.info("No retention action recommended (insufficient signals or unknown customer).")

        # 3. Drivers table
        st.markdown("**Drivers**")
        drivers = resp.get("drivers", [])
        if drivers:
            rows = [
                {
                    "Driver": d["driver"].replace("_", " ").title(),
                    "Evidence": d["evidence"],
                    "Source": ", ".join(d["citations"]),
                }
                for d in drivers
            ]
            st.table(rows)
        else:
            st.caption("No verified dissatisfaction drivers.")

        if risk and risk.get("signals_used"):
            weight_parts = [
                f"{sig} (+{SIGNAL_WEIGHTS.get(sig, 0)})" for sig in risk["signals_used"]
            ]
            st.caption("Score: " + ", ".join(weight_parts))

        if resp.get("data_conflicts"):
            for conflict in resp["data_conflicts"]:
                st.warning(f"Data conflict: {conflict}")

        if resp.get("missing_information"):
            st.warning("Missing: " + "; ".join(resp["missing_information"]))

        if resp.get("unsupported"):
            st.info("Not supported: " + "; ".join(resp["unsupported"]))

        # Escalation / Supervisor review status & HITL controls
        if resp.get("escalate"):
            st.error(f"🟠 Supervisor review required: {resp.get('escalation_reason')}")
            if res["status"] == "ESCALATED_PENDING_SUPERVISOR":
                with st.form("supervisor_form"):
                    st.markdown("**Supervisor Decision (LangGraph Interrupt)**")
                    rev = st.text_input("Supervisor Name", value="Duty Supervisor")
                    notes = st.text_input("Supervisor Comments", value="")
                    c1, c2 = st.columns(2)
                    app_btn = c1.form_submit_button("✅ Approve Exception", use_container_width=True)
                    rej_btn = c2.form_submit_button("❌ Reject Exception", use_container_width=True)
                if app_btn:
                    st.session_state.churn_result = service.resolve_supervisor_review(
                        res["thread_id"], "approve", reviewer=rev, comments=notes
                    )
                    st.rerun()
                elif rej_btn:
                    st.session_state.churn_result = service.resolve_supervisor_review(
                        res["thread_id"], "reject", reviewer=rev, comments=notes
                    )
                    st.rerun()
            else:
                st.success("✅ Supervisor review completed.")
        else:
            st.markdown("🟢 **Supervisor review:** not needed")

        with st.expander("Summary & Raw JSON"):
            st.write(resp.get("summary", ""))
            st.json(resp)

        with st.expander("LangGraph 8-Node Execution Trace"):
            for idx, step in enumerate(res.get("trace", []), start=1):
                st.markdown(f"`{idx}.` {step}")
