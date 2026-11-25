"""Pydantic schemas and LangGraph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

DRIVER_ORDER: tuple[str, ...] = (
    "network_quality",
    "billing_dispute",
    "plan_fit",
    "service_experience",
    "competitor_intent",
    "usage_decline",
)

ACTION_CATALOG: tuple[str, ...] = (
    "plan_review",
    "network_escalation",
    "bill_correction",
    "callback",
)

DriverName = Literal[
    "network_quality",
    "billing_dispute",
    "plan_fit",
    "service_experience",
    "competitor_intent",
    "usage_decline",
]

ActionName = Literal[
    "plan_review",
    "network_escalation",
    "bill_correction",
    "callback",
]

RiskLevel = Literal["low", "medium", "high"]


class RequestContext(BaseModel):
    channel: str = Field(default="inbound_call")
    stated_reason: str = Field(default="")


class InsightRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    question: str = Field(default="")
    context: RequestContext = Field(default_factory=RequestContext)
    session_date: Optional[str] = Field(
        default=None,
        description="Optional ISO date YYYY-MM-DD for deterministic policy validity evaluation.",
    )


class DriverItem(BaseModel):
    driver: DriverName
    evidence: str
    citations: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    level: RiskLevel
    score: int = Field(..., ge=0)
    signals_used: list[str] = Field(default_factory=list)


class RecommendedAction(BaseModel):
    action: ActionName
    reasoning: str
    based_on_signals: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    follow_up: Optional[str] = None


class InsightResponse(BaseModel):
    customer_id: str
    summary: str
    drivers: list[DriverItem] = Field(default_factory=list)
    risk: Optional[RiskAssessment] = None
    recommended_action: Optional[RecommendedAction] = None
    missing_information: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    data_conflicts: list[str] = Field(default_factory=list)
    escalate: bool = False
    escalation_reason: Optional[str] = None


class ChurnGraphState(TypedDict, total=False):
    thread_id: str
    session_date: str
    raw_request: dict[str, Any]
    sanitized_request: dict[str, Any]
    pii_redactions: list[str]
    known_customer: bool
    customer_profile: dict[str, Any]
    verified_records: list[dict[str, Any]]
    unverified_record_ids: list[str]
    active_policies: list[dict[str, Any]]
    expired_policies: list[dict[str, Any]]
    data_conflicts: list[str]
    has_unresolved_conflict: bool
    unsupported_items: list[str]
    missing_information: list[str]
    policy_exception_requested: bool
    policy_exception_reason: Optional[str]
    risk: Optional[dict[str, Any]]
    draft_response: dict[str, Any]
    escalate: bool
    escalation_reason: Optional[str]
    supervisor_decision: Optional[str]
    supervisor_reviewer: Optional[str]
    supervisor_comments: Optional[str]
    final_response: dict[str, Any]
    execution_trace: Annotated[list[str], operator.add]
