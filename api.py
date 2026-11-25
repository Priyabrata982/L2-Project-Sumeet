"""FastAPI Churn Insight API (POST /insight + Supervisor resume endpoint)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_assistant.graph import ChurnInsightService
from churn_assistant.schemas import InsightRequest, InsightResponse

app = FastAPI(
    title="Telecom Customer Churn Insight Assistant API",
    version="1.0.0",
    description="LangGraph + Ollama churn insight service with code-enforced guardrails.",
)

service = ChurnInsightService()


class SupervisorDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = "Supervisor"
    comments: str = ""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/insight", response_model=InsightResponse)
async def post_insight(payload: InsightRequest) -> InsightResponse:
    """Analyze customer churn drivers, risk band, and within-policy retention action."""
    result = service.analyze(payload.model_dump())
    return InsightResponse(**result["response"])


@app.post("/insight/with-thread")
async def post_insight_with_thread(payload: InsightRequest) -> dict:
    """Run insight analysis and return thread_id + status alongside the InsightResponse."""
    return service.analyze(payload.model_dump())


@app.post("/insight/{thread_id}/supervisor")
async def resolve_supervisor(
    thread_id: str, body: SupervisorDecisionRequest
) -> dict:
    """Resume a paused escalation at the Supervisor review node."""
    try:
        return service.resolve_supervisor_review(
            thread_id=thread_id,
            decision=body.decision,
            reviewer=body.reviewer,
            comments=body.comments,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
