"""8-node LangGraph workflow matching the Reference Architecture (Page 3)."""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import settings
from .guards import run_guard_checks
from .llm_provider import generate_insight_draft
from .pii import mask_personal_data
from .retrieval import DataPackRepository
from .schemas import ChurnGraphState, InsightRequest, InsightResponse, RiskAssessment
from .scoring import compute_churn_risk


def build_churn_graph(
    repo: Optional[DataPackRepository] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """Compile the 8-node LangGraph workflow with 1 escalation decision."""
    data_repo = repo or DataPackRepository()
    memory = checkpointer or MemorySaver()

    # Node 1: Request
    def request_node(state: ChurnGraphState) -> dict[str, Any]:
        req = InsightRequest(**state["raw_request"])
        session_date = req.session_date or state.get("session_date") or settings.default_session_date
        return {
            "raw_request": req.model_dump(),
            "session_date": session_date,
            "execution_trace": [f"Request -> customer_id={req.customer_id}"],
        }

    # Node 2: Check input (known customer ID + mask personal data)
    def check_input_node(state: ChurnGraphState) -> dict[str, Any]:
        req = state["raw_request"]
        cid = str(req.get("customer_id", "")).strip().upper()
        masked_q, red_q = mask_personal_data(req.get("question", ""))
        ctx = req.get("context", {}) or {}
        masked_reason, red_r = mask_personal_data(ctx.get("stated_reason", ""))
        redactions = list(dict.fromkeys(red_q + red_r))

        sanitized = {
            "customer_id": cid,
            "question": masked_q,
            "context": {
                "channel": ctx.get("channel", "inbound_call"),
                "stated_reason": masked_reason,
            },
        }
        known = cid in data_repo.customers
        return {
            "sanitized_request": sanitized,
            "pii_redactions": redactions,
            "known_customer": known,
            "execution_trace": [
                f"Check input -> known={known}, pii_masked={redactions}"
            ],
        }

    # Node 3: Retrieve records (verified history + active/expired policies + conflicts)
    def retrieve_records_node(state: ChurnGraphState) -> dict[str, Any]:
        cid = state["sanitized_request"]["customer_id"]
        session_date = state["session_date"]
        bundle = data_repo.retrieve(cid, session_date)
        return {
            "customer_profile": bundle.customer_profile,
            "verified_records": bundle.verified_records,
            "unverified_record_ids": bundle.unverified_record_ids,
            "active_policies": bundle.active_policies,
            "expired_policies": bundle.expired_policies,
            "data_conflicts": bundle.data_conflicts,
            "has_unresolved_conflict": bundle.has_unresolved_conflict,
            "execution_trace": [
                f"Retrieve records -> verified={len(bundle.verified_records)}, "
                f"unverified_excluded={len(bundle.unverified_record_ids)}, "
                f"conflicts={len(bundle.data_conflicts)}"
            ],
        }

    # Node 4: Score risk (fixed signal-scoring rule in code)
    def score_risk_node(state: ChurnGraphState) -> dict[str, Any]:
        if not state.get("known_customer"):
            return {
                "risk": None,
                "execution_trace": ["Score risk -> skipped (unknown customer)"],
            }
        risk = compute_churn_risk(
            verified_records=state.get("verified_records", []),
            session_date=state["session_date"],
        )
        return {
            "risk": risk.model_dump() if risk else None,
            "execution_trace": [
                f"Score risk -> {risk.level} ({risk.score})"
                if risk
                else "Score risk -> null (no signals)"
            ],
        }

    # Node 5: Generate (summary, drivers, action from catalog)
    def generate_node(state: ChurnGraphState) -> dict[str, Any]:
        san = state["sanitized_request"]
        risk_dict = state.get("risk")
        risk_obj = RiskAssessment(**risk_dict) if risk_dict else None
        draft, backend = generate_insight_draft(
            customer_id=san["customer_id"],
            known_customer=bool(state.get("known_customer")),
            customer_profile=state.get("customer_profile", {}),
            question=san.get("question", ""),
            stated_reason=san.get("context", {}).get("stated_reason", ""),
            session_date=state["session_date"],
            verified_records=state.get("verified_records", []),
            active_policies=state.get("active_policies", []),
            expired_policies=state.get("expired_policies", []),
            data_conflicts=state.get("data_conflicts", []),
            risk=risk_obj,
        )
        return {
            "draft_response": draft.model_dump(),
            "execution_trace": [f"Generate ({backend}) -> draft built"],
        }

    # Node 6: Guard checks (citations resolve, policy limits, conflicts, escalate flag)
    def guard_checks_node(state: ChurnGraphState) -> dict[str, Any]:
        san = state["sanitized_request"]
        draft = InsightResponse(**state["draft_response"])
        risk_dict = state.get("risk")
        computed_risk = RiskAssessment(**risk_dict) if risk_dict else None

        checked = run_guard_checks(
            draft=draft,
            known_customer=bool(state.get("known_customer")),
            question=san.get("question", ""),
            stated_reason=san.get("context", {}).get("stated_reason", ""),
            verified_records=state.get("verified_records", []),
            unverified_record_ids=state.get("unverified_record_ids", []),
            active_policies=state.get("active_policies", []),
            expired_policies=state.get("expired_policies", []),
            data_conflicts=state.get("data_conflicts", []),
            has_unresolved_conflict=bool(state.get("has_unresolved_conflict")),
            computed_risk=computed_risk,
        )
        return {
            "draft_response": checked.model_dump(),
            "escalate": checked.escalate,
            "escalation_reason": checked.escalation_reason,
            "execution_trace": [
                f"Guard checks -> escalate={checked.escalate}"
            ],
        }

    # Escalation Decision (Escalate?)
    def route_escalate(
        state: ChurnGraphState,
    ) -> Literal["Supervisor review", "Response"]:
        if state.get("escalate"):
            return "Supervisor review"
        return "Response"

    # Node 7: Supervisor review (interrupt, then approve or reject)
    def supervisor_review_node(state: ChurnGraphState) -> dict[str, Any]:
        draft = state["draft_response"]
        decision_payload = interrupt(
            {
                "interrupt_type": "supervisor_review",
                "customer_id": draft["customer_id"],
                "escalation_reason": state.get("escalation_reason"),
                "draft_response": draft,
                "message": "Policy exception, unresolved conflict, or citation failure requires supervisor approval or rejection.",
            }
        )

        decision = "reject"
        reviewer = "Supervisor"
        comments = ""
        if isinstance(decision_payload, dict):
            decision = str(decision_payload.get("decision", "reject")).strip().lower()
            reviewer = str(decision_payload.get("reviewer") or "Supervisor").strip()
            comments = str(decision_payload.get("comments") or "").strip()

        normalized = "approve" if decision in ("approve", "approved", "yes") else "reject"
        updated_draft = dict(draft)
        if normalized == "approve":
            updated_draft["summary"] = (
                f"{updated_draft['summary']} [Supervisor ({reviewer}) approved exception"
                + (f": {comments}]" if comments else "]")
            )
        else:
            updated_draft["summary"] = (
                f"{updated_draft['summary']} [Supervisor ({reviewer}) rejected exception; "
                "only standard RET-POL catalog limits apply"
                + (f": {comments}]" if comments else "]")
            )

        return {
            "draft_response": updated_draft,
            "supervisor_decision": normalized,
            "supervisor_reviewer": reviewer,
            "supervisor_comments": comments,
            "execution_trace": [
                f"Supervisor review -> {normalized.upper()} by {reviewer}"
            ],
        }

    # Node 8: Response (final structured output to agent UI / API)
    def response_node(state: ChurnGraphState) -> dict[str, Any]:
        final = InsightResponse(**state["draft_response"]).model_dump()
        return {
            "final_response": final,
            "execution_trace": [
                f"Response -> level={(final.get('risk') or {}).get('level')}, escalate={final.get('escalate')}"
            ],
        }

    builder = StateGraph(ChurnGraphState)
    builder.add_node("Request", request_node)
    builder.add_node("Check input", check_input_node)
    builder.add_node("Retrieve records", retrieve_records_node)
    builder.add_node("Score risk", score_risk_node)
    builder.add_node("Generate", generate_node)
    builder.add_node("Guard checks", guard_checks_node)
    builder.add_node("Supervisor review", supervisor_review_node)
    builder.add_node("Response", response_node)

    builder.add_edge(START, "Request")
    builder.add_edge("Request", "Check input")
    builder.add_edge("Check input", "Retrieve records")
    builder.add_edge("Retrieve records", "Score risk")
    builder.add_edge("Score risk", "Generate")
    builder.add_edge("Generate", "Guard checks")
    builder.add_conditional_edges("Guard checks", route_escalate)
    builder.add_edge("Supervisor review", "Response")
    builder.add_edge("Response", END)

    return builder.compile(checkpointer=memory)


class ChurnInsightService:
    """Service wrapper for running and resuming Telecom Churn Insight workflows."""

    def __init__(self, repo: Optional[DataPackRepository] = None):
        self.repo = repo or DataPackRepository()
        self.checkpointer = MemorySaver()
        self.graph = build_churn_graph(repo=self.repo, checkpointer=self.checkpointer)

    def _format_output(self, thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        interrupts: list[Any] = []
        if snapshot.tasks:
            for task in snapshot.tasks:
                if getattr(task, "interrupts", None):
                    interrupts.extend(item.value for item in task.interrupts)

        if interrupts:
            intr = interrupts[0]
            return {
                "status": "ESCALATED_PENDING_SUPERVISOR",
                "thread_id": thread_id,
                "interrupt": intr,
                "response": intr["draft_response"],
                "trace": snapshot.values.get("execution_trace", []),
            }

        return {
            "status": "COMPLETED",
            "thread_id": thread_id,
            "response": result["final_response"],
            "trace": result.get("execution_trace", []),
        }

    def analyze(
        self, request_payload: dict[str, Any], thread_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Run the 8-node LangGraph insight pipeline."""
        tid = thread_id or f"churn-{uuid4().hex[:10]}"
        config = {"configurable": {"thread_id": tid}}
        initial: ChurnGraphState = {
            "thread_id": tid,
            "raw_request": request_payload,
            "execution_trace": [],
        }
        result = self.graph.invoke(initial, config=config)
        return self._format_output(tid, result)

    def resolve_supervisor_review(
        self,
        thread_id: str,
        decision: Literal["approve", "reject"],
        reviewer: str = "Supervisor",
        comments: str = "",
    ) -> dict[str, Any]:
        """Resume a paused graph at Supervisor review with approve/reject."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "reviewer": reviewer,
                    "comments": comments,
                }
            ),
            config=config,
        )
        return self._format_output(thread_id, result)
